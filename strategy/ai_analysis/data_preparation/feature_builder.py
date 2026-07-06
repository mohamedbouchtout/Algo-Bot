"""
Combines the individual feature extractors into a model-ready dataset.

Pipeline
--------
1. Run every registered extractor on one ticker's OHLCV bars.
2. Concatenate the results into a single continuous feature matrix.
3. Optionally fit robust per-feature bin edges (quantile based) for RBM
   binarization (legacy, disabled by default).
4. Build sliding windows of length `window_size` and flatten them.
5. Generate volatility-adjusted labels from forward returns.
"""

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy.ai_analysis.data_preparation.indicator_features import IndicatorFeatureExtractor
from strategy.ai_analysis.data_preparation.market_features import MarketFeatureExtractor
from strategy.ai_analysis.data_preparation.price_features import PriceFeatureExtractor
from strategy.ai_analysis.data_preparation.volume_features import VolumeFeatureExtractor

logger = logging.getLogger(__name__)


class FeatureBuilder:
    def __init__(
        self,
        window_size: int = 10,
        n_bits: int = 4,
        extractors: list | None = None,
        forward_horizon: int = 5,
        label_threshold: float = 0.01,
        volatility_adjusted_labels: bool = True,
        volatility_threshold: float = 1.0,
    ):
        """
        Parameters
        ----------
        window_size : how many consecutive days of features form one training sample.
        n_bits      : thermometer-encoding resolution per continuous feature (legacy RBM).
        extractors  : list of feature extractor instances (must expose .extract(df)
                      returning a DataFrame and .FEATURE_NAMES). Defaults to
                      the four standard ones (price, volume, indicator, market).
        forward_horizon : how many days ahead the label looks.
        label_threshold : forward return magnitude for fixed-threshold labels.
        volatility_adjusted_labels : if True, normalize forward returns by ATR
                                     before applying the threshold.
        volatility_threshold : ATR-normalized return threshold (used when
                               volatility_adjusted_labels is True).
        """
        self.window_size = window_size
        self.n_bits = n_bits
        self.extractors = extractors or [
            PriceFeatureExtractor(),
            VolumeFeatureExtractor(),
            IndicatorFeatureExtractor(),
            MarketFeatureExtractor(),
        ]
        self.forward_horizon = forward_horizon
        self.label_threshold = label_threshold
        self.volatility_adjusted_labels = volatility_adjusted_labels
        self.volatility_threshold = volatility_threshold

        self.feature_names: list[str] = []
        self.bin_edges: dict[str, np.ndarray] = {}
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None

    def build_continuous_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run every extractor on one ticker's bars and concat side-by-side."""
        frames = [ex.extract(df) for ex in self.extractors]
        combined = pd.concat(frames, axis=1)
        return combined

    def fit_bin_edges(self, per_ticker_frames: Iterable[pd.DataFrame]) -> None:
        """
        Learn quantile based bin edges for every feature from the entire
        training corpus. Must be called once before binarize().
        """
        pooled = pd.concat(list(per_ticker_frames), axis=0, ignore_index=True)
        pooled = pooled.replace([np.inf, -np.inf], np.nan).dropna(how='any')

        if pooled.empty:
            raise ValueError('No clean rows to fit bin edges, check input data')

        self.feature_names = list(pooled.columns)
        quantiles = np.linspace(0.0, 1.0, self.n_bits + 2)[1:-1]
        self.bin_edges = {col: np.quantile(pooled[col].values, quantiles) for col in self.feature_names}

        self._feat_mean = pooled[self.feature_names].mean().values.astype(np.float32)
        self._feat_std = pooled[self.feature_names].std().values.astype(np.float32)
        self._feat_std[self._feat_std < 1e-8] = 1.0

        logger.info(f'FeatureBuilder fit: {len(self.feature_names)} features, {self.n_bits} bits/feature, pooled rows={len(pooled)}')

    def binarize(self, features: pd.DataFrame) -> np.ndarray:
        """
        Thermometer-encode a continuous feature DataFrame.
        Returns a (rows, n_features * n_bits) uint8 array.
        """
        if not self.bin_edges:
            raise RuntimeError('fit_bin_edges() must be called before binarize()')

        arrs = []
        for col in self.feature_names:
            edges = self.bin_edges[col]
            vals = features[col].values[:, None]
            arrs.append((vals > edges[None, :]).astype(np.uint8))
        return np.concatenate(arrs, axis=1)

    def build_windows(
        self,
        df: pd.DataFrame,
        include_labels: bool = True,
        include_rbm: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Build sliding window samples for one ticker.

        Returns
        -------
        rbm_x  : binarized windows (empty if include_rbm=False)
        cnn_x  : (n_samples, window_size * n_features) float32
        labels : (n_samples,) int {0: short, 1: flat, 2: long} or None
        """
        features = self.build_continuous_features(df)
        features = features.replace([np.inf, -np.inf], np.nan)

        close = df['close'].astype(float).values if 'close' in df.columns else None

        valid_mask = features.notna().all(axis=1).values
        features = features[valid_mask].reset_index(drop=True)
        if close is not None:
            close = close[valid_mask]

        if not self.feature_names:
            self.feature_names = list(features.columns)

        n_features = len(self.feature_names) if self.feature_names else len(features.columns)

        if len(features) <= self.window_size + (self.forward_horizon if include_labels else 0):
            return (
                np.empty((0, self.window_size * n_features * self.n_bits), dtype=np.uint8),
                np.empty((0, self.window_size * n_features), dtype=np.float32),
                None if not include_labels else np.empty((0,), dtype=np.int64),
            )

        cont_raw = features[self.feature_names].values.astype(np.float32) if self.feature_names else features.values.astype(np.float32)

        if self._feat_mean is not None and self._feat_std is not None:
            cont = (cont_raw - self._feat_mean) / self._feat_std
        else:
            cont = cont_raw

        if include_rbm and self.bin_edges:
            bits = self.binarize(features)
        else:
            bits = None

        T = len(features)
        last_start = T - self.window_size - (self.forward_horizon if include_labels else 0)
        starts = np.arange(0, last_start)

        if bits is not None:
            rbm_x = np.stack([bits[s : s + self.window_size].reshape(-1) for s in starts]).astype(np.uint8)
        else:
            rbm_x = np.empty((len(starts), 0), dtype=np.uint8)

        cnn_x = np.stack([cont[s : s + self.window_size].reshape(-1) for s in starts])

        labels = None
        if include_labels and close is not None:
            end_idx = starts + self.window_size - 1
            fwd_idx = end_idx + self.forward_horizon
            fwd_return = (close[fwd_idx] / close[end_idx]) - 1.0

            if self.volatility_adjusted_labels and 'atr14_pct' in self.feature_names:
                atr_col_idx = self.feature_names.index('atr14_pct')
                atr_values = cont_raw[end_idx, atr_col_idx]
                safe_atr = np.where(
                    (atr_values > 0) & np.isfinite(atr_values),
                    atr_values,
                    self.label_threshold,
                )
                adjusted_return = fwd_return / safe_atr
                labels = np.full(len(starts), 1, dtype=np.int64)
                labels[adjusted_return > self.volatility_threshold] = 2
                labels[adjusted_return < -self.volatility_threshold] = 0
            else:
                labels = np.full(len(starts), 1, dtype=np.int64)
                labels[fwd_return > self.label_threshold] = 2
                labels[fwd_return < -self.label_threshold] = 0

        return rbm_x, cnn_x, labels

    @property
    def visible_dim(self) -> int:
        """visible_dim to pass to the RBM constructor (legacy)."""
        return self.window_size * len(self.feature_names) * self.n_bits

    @property
    def cnn_input_length(self) -> int:
        """Length of the flattened CNN/LSTM input signal."""
        return self.window_size * len(self.feature_names)
