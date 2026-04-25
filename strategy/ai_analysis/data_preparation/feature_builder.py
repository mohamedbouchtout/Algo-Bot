"""
Combines the individual feature extractors into a model-ready dataset.

Pipeline
--------
1. Run every registered extractor on one ticker's OHLCV bars.
2. Concatenate the results into a single continuous feature matrix.
3. Fit robust per-feature bin edges (quantile based) across the full training
   corpus. These are then reused to transform any new data consistently.
4. Encode each continuous feature with thermometer encoding into `n_bits`
   binary values so that the result is suitable as the visible layer of a
   Bernoulli RBM.
5. Build sliding windows of length `window_size` and flatten them, producing
   the final (samples, window_size * total_bits) matrix.

The same continuous (non-binarised) windowed matrix is also returned so the
CNN can consume it directly as a 1-D signal.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Optional, Tuple

from strategy.ai_analysis.data_preparation.price_features import PriceFeatureExtractor
from strategy.ai_analysis.data_preparation.volume_features import VolumeFeatureExtractor
from strategy.ai_analysis.data_preparation.indicator_features import IndicatorFeatureExtractor

logger = logging.getLogger(__name__)


class FeatureBuilder:
    def __init__(
        self,
        window_size: int = 10,
        n_bits: int = 4,
        extractors: Optional[List] = None,
        forward_horizon: int = 5,
        label_threshold: float = 0.01,
    ):
        """
        Parameters
        ----------
        window_size : how many consecutive days of features form one training sample.
        n_bits      : thermometer-encoding resolution per continuous feature.
                      visible_dim of RBM  == window_size * sum(extractor feature counts) * n_bits.
        extractors  : list of feature extractor instances (must expose .extract(df)
                      returning a DataFrame and .FEATURE_NAMES). Defaults to
                      the three standard ones.
        forward_horizon : how many days ahead the CNN label looks.
        label_threshold : forward return magnitude that separates LONG / SHORT / FLAT.
        """
        self.window_size = window_size
        self.n_bits = n_bits
        self.extractors = extractors or [
            PriceFeatureExtractor(),
            VolumeFeatureExtractor(),
            IndicatorFeatureExtractor(),
        ]
        self.forward_horizon = forward_horizon
        self.label_threshold = label_threshold

        # Populated by fit_bin_edges()
        self.feature_names: List[str] = []
        self.bin_edges: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ public
    def build_continuous_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run every extractor on one ticker's bars and concat side-by-side."""
        frames = [ex.extract(df) for ex in self.extractors]
        combined = pd.concat(frames, axis=1)
        return combined

    def fit_bin_edges(self, per_ticker_frames: Iterable[pd.DataFrame]) -> None:
        """
        Learn quantile based bin edges for every feature from the entire
        training corpus. Must be called once before transform().
        """
        pooled = pd.concat(list(per_ticker_frames), axis=0, ignore_index=True)
        pooled = pooled.replace([np.inf, -np.inf], np.nan).dropna(how='any')

        if pooled.empty:
            raise ValueError("No clean rows to fit bin edges, check input data")

        self.feature_names = list(pooled.columns)
        # n_bits thresholds -> splits the distribution into n_bits+1 buckets.
        # Thermometer encoding of bucket k turns on the first k bits.
        quantiles = np.linspace(0.0, 1.0, self.n_bits + 2)[1:-1]
        self.bin_edges = {
            col: np.quantile(pooled[col].values, quantiles)
            for col in self.feature_names
        }
        logger.info(
            f"FeatureBuilder fit: {len(self.feature_names)} features, "
            f"{self.n_bits} bits/feature, pooled rows={len(pooled)}"
        )

    def binarize(self, features: pd.DataFrame) -> np.ndarray:
        """
        Thermometer-encode a continuous feature DataFrame.

        Returns a (rows, n_features * n_bits) uint8 array.
        """
        if not self.bin_edges:
            raise RuntimeError("fit_bin_edges() must be called before binarize()")

        arrs = []
        for col in self.feature_names:
            edges = self.bin_edges[col]               # shape (n_bits,)
            vals = features[col].values[:, None]      # shape (rows, 1)
            # bit k on when value > edge[k]  -> thermometer encoding
            arrs.append((vals > edges[None, :]).astype(np.uint8))
        return np.concatenate(arrs, axis=1)

    def build_windows(
        self,
        df: pd.DataFrame,
        include_labels: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Build sliding window samples for one ticker.

        Returns
        -------
        rbm_x  : (n_samples, window_size * total_bits) uint8, for the RBM
        cnn_x  : (n_samples, window_size * n_features) float32, continuous, for the CNN
        labels : (n_samples,) int {0: short, 1: flat, 2: long}  or None
        """
        features = self.build_continuous_features(df)
        features = features.replace([np.inf, -np.inf], np.nan)

        close = df['close'].astype(float).values if 'close' in df.columns else None

        # Drop leading NaNs caused by the rolling windows in the extractors
        valid_mask = features.notna().all(axis=1).values
        features = features[valid_mask].reset_index(drop=True)
        if close is not None:
            close = close[valid_mask]

        if len(features) <= self.window_size + self.forward_horizon:
            return (
                np.empty((0, self.window_size * len(self.feature_names) * self.n_bits), dtype=np.uint8),
                np.empty((0, self.window_size * len(self.feature_names)), dtype=np.float32),
                None if not include_labels else np.empty((0,), dtype=np.int64),
            )

        # Binarise once for the whole ticker, then slide
        bits = self.binarize(features)                        # (T, F*B)
        cont = features[self.feature_names].values.astype(np.float32)  # (T, F)

        T = len(features)
        last_start = T - self.window_size - (self.forward_horizon if include_labels else 0)
        starts = np.arange(0, last_start)

        rbm_x = np.stack([bits[s:s + self.window_size].reshape(-1) for s in starts])
        cnn_x = np.stack([cont[s:s + self.window_size].reshape(-1) for s in starts])

        labels = None
        if include_labels and close is not None:
            end_idx = starts + self.window_size - 1        # index of the last bar in the window
            fwd_idx = end_idx + self.forward_horizon       # index `forward_horizon` bars later
            fwd_return = (close[fwd_idx] / close[end_idx]) - 1.0

            labels = np.full(len(starts), 1, dtype=np.int64)   # default = flat
            labels[fwd_return > self.label_threshold] = 2      # long
            labels[fwd_return < -self.label_threshold] = 0     # short

        return rbm_x.astype(np.uint8), cnn_x, labels

    # -------------------------------------------------------------- sizes
    @property
    def visible_dim(self) -> int:
        """visible_dim to pass to the RBM constructor."""
        return self.window_size * len(self.feature_names) * self.n_bits

    @property
    def cnn_input_length(self) -> int:
        """Length of the 1D CNN input signal."""
        return self.window_size * len(self.feature_names)
