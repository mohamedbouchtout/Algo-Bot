"""
High-level orchestrator for the AI analysis pipeline.

Responsibilities
----------------
* Pull historical data for a list of tickers via the existing StockDataFetcher.
* Build a combined, binarised training set with FeatureBuilder.
* Train the RBM on the full corpus (unsupervised).
* Use the trained RBM to produce hidden features for every sample.
* Train the CNN on (continuous window, RBM features) -> forward-return label.
* Expose `predict(symbol)` that pulls the latest bars and returns a LONG/FLAT/
  SHORT classification plus class probabilities, this is the hook that
  `order_manager.scan_stocks()` can eventually call.

Two training styles are supported:

1. Batch ::
       analyzer.train(['AAPL', 'MSFT', ...])
   Pulls every ticker, pools their windows, fits the models once.

2. Incremental (one ticker per call) ::
       for sym in tickers:
           analyzer.add_ticker(sym)
       analyzer.finalize_training()
   Fits the same pooled models, but lets the caller drive the per-ticker
   loop, useful when integrating with an existing scan loop and when you
   want to skip / retry individual tickers.

Both styles produce equivalent models; training a model on a single ticker
in isolation is **not** recommended because the RBM/CNN need the pooled
cross-ticker samples to learn general patterns rather than memorising one
name.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_fetch.historical_data import StockDataFetcher
from strategy.ai_analysis.data_preparation.feature_builder import FeatureBuilder
from strategy.ai_analysis.rbm_trainer import RBMTrainer
from strategy.ai_analysis.cnn_trainer import CNNTrainer

logger = logging.getLogger(__name__)


class AIAnalyzer:
    CLASS_NAMES = {0: 'SHORT', 1: 'FLAT', 2: 'LONG'}

    def __init__(
        self,
        stock_data: StockDataFetcher,
        feature_builder: Optional[FeatureBuilder] = None,
        rbm_hidden_dim: int = 64,
        rbm_epochs: int = 30,
        cnn_epochs: int = 20,
    ):
        self.stock_data = stock_data
        self.feature_builder = feature_builder or FeatureBuilder(window_size=10, n_bits=4)
        self.rbm_trainer: Optional[RBMTrainer] = None
        self.cnn_trainer: Optional[CNNTrainer] = None
        self.rbm_epochs = rbm_epochs
        self.cnn_epochs = cnn_epochs
        self.rbm_hidden_dim = rbm_hidden_dim

        # Cache raw bars per ticker so we don't hit IB twice when building the
        # dataset and then the labelled samples.
        self._bar_cache: Dict[str, pd.DataFrame] = {}

        # Per-ticker continuous features accumulated across add_ticker() calls.
        # These get concatenated into the bin-edges fit in finalize_training().
        self._continuous_per_ticker: List[pd.DataFrame] = []
        self._kept_tickers: List[str] = []

    # =================================================================== data
    def _get_bars(self, symbol: str) -> Optional[pd.DataFrame]:
        if symbol in self._bar_cache:
            return self._bar_cache[symbol]
        df = self.stock_data.get_historical_data(symbol)
        if df is not None:
            self._bar_cache[symbol] = df
        return df

    def reset_dataset(self) -> None:
        """Drop accumulated bars / features so the next add_ticker() starts fresh."""
        self._bar_cache.clear()
        self._continuous_per_ticker.clear()
        self._kept_tickers.clear()

    def add_ticker(self, symbol: str) -> bool:
        """
        Incrementally add one ticker's data to the training corpus.

        Fetches bars (if not already cached), computes continuous features and
        stores them for the next `finalize_training()` call. Does **not**
        train anything yet.

        Returns
        -------
        True  if the ticker was added.
        False if it was skipped (insufficient data / extraction error / already added).
        """
        if symbol in self._kept_tickers:
            logger.debug(f"{symbol}: already in dataset, skipping")
            return False

        bars = self._get_bars(symbol)
        if bars is None or len(bars) < 250:
            logger.info(f"Skipping {symbol}: insufficient bars")
            return False

        try:
            feats = self.feature_builder.build_continuous_features(bars)
        except ValueError as e:
            logger.warning(f"{symbol}: {e}")
            return False

        self._continuous_per_ticker.append(feats)
        self._kept_tickers.append(symbol)
        logger.debug(
            f"Added {symbol} to dataset "
            f"({len(self._kept_tickers)} tickers accumulated)"
        )
        return True

    def build_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Assemble the pooled training tensors from whatever has been accumulated
        via `add_ticker()`.

        Returns
        -------
        rbm_x  : (N, visible_dim) uint8     binarised windows for the RBM
        cnn_x  : (N, input_length) float32  continuous windows for the CNN
        labels : (N,) int64                 0/1/2 target classes
        ids    : (N,) int64                 per-sample ticker index for diagnostics.
        """
        if not self._continuous_per_ticker:
            raise RuntimeError(
                "No tickers in dataset, call add_ticker() (or train(tickers)) first"
            )

        self.feature_builder.fit_bin_edges(self._continuous_per_ticker)

        rbm_chunks, cnn_chunks, label_chunks, ticker_ids = [], [], [], []
        for idx, sym in enumerate(self._kept_tickers):
            bars = self._bar_cache[sym]
            rbm_x, cnn_x, labels = self.feature_builder.build_windows(bars, include_labels=True)
            if len(rbm_x) == 0:
                continue
            rbm_chunks.append(rbm_x)
            cnn_chunks.append(cnn_x)
            label_chunks.append(labels)
            ticker_ids.append(np.full(len(rbm_x), idx, dtype=np.int64))

        if not rbm_chunks:
            raise RuntimeError("No tickers produced usable windowed samples")

        rbm_all = np.concatenate(rbm_chunks, axis=0)
        cnn_all = np.concatenate(cnn_chunks, axis=0)
        labels_all = np.concatenate(label_chunks, axis=0)
        ids_all = np.concatenate(ticker_ids, axis=0)

        logger.info(
            f"Dataset built: {len(rbm_all)} samples across {len(self._kept_tickers)} tickers "
            f"(visible_dim={rbm_all.shape[1]}, cnn_len={cnn_all.shape[1]}, "
            f"class counts={np.bincount(labels_all, minlength=3).tolist()})"
        )
        return rbm_all, cnn_all, labels_all, ids_all

    # ================================================================== train
    def finalize_training(self, val_split: float = 0.2) -> None:
        """
        Fit the RBM and CNN on everything accumulated via `add_ticker()`.
        Training on a single ticker is technically allowed but strongly
        discouraged (models will just memorise that ticker).
        """
        if len(self._kept_tickers) < 2:
            logger.warning(
                f"finalize_training() called with only {len(self._kept_tickers)} "
                "ticker(s); pooled training needs several tickers to generalise."
            )

        rbm_x, cnn_x, labels, _ = self.build_dataset()

        split = int(len(rbm_x) * (1.0 - val_split))
        x_train, x_test = rbm_x[:split], rbm_x[split:]

        # --- RBM ---------------------------------------------------------
        self.rbm_trainer = RBMTrainer(
            visible_dim=self.feature_builder.visible_dim,
            hidden_dim=self.rbm_hidden_dim,
            epochs=self.rbm_epochs,
        )
        self.rbm_trainer.train(x_train, x_test)

        # --- CNN ---------------------------------------------------------
        rbm_feats = self.rbm_trainer.hidden_features(rbm_x)
        self.cnn_trainer = CNNTrainer(
            input_length=self.feature_builder.cnn_input_length,
            rbm_feature_dim=self.rbm_hidden_dim,
            epochs=self.cnn_epochs,
        )
        self.cnn_trainer.train(cnn_x, rbm_feats, labels, val_split=val_split)

    def train(self, tickers: List[str], val_split: float = 0.2) -> None:
        """
        Convenience wrapper: accumulate every ticker in `tickers`, then fit.
        Equivalent to calling `add_ticker()` in a loop and then
        `finalize_training()`.
        """
        for sym in tickers:
            self.add_ticker(sym)
        self.finalize_training(val_split=val_split)

    # ================================================================ predict
    def predict(self, symbol: str) -> Optional[Dict]:
        """
        Fetch the latest bars for `symbol`, build the most recent window and
        classify it.  Returns None if anything is missing / not trained.
        """
        if self.rbm_trainer is None or self.cnn_trainer is None:
            raise RuntimeError("Call train() or finalize_training() before predict()")

        df = self.stock_data.get_historical_data(symbol)
        if df is None or len(df) < 250:
            return None

        rbm_x, cnn_x, _ = self.feature_builder.build_windows(df, include_labels=False)
        if len(rbm_x) == 0:
            return None

        # Use only the most recent window
        rbm_last = rbm_x[-1:]
        cnn_last = cnn_x[-1:]
        rbm_feats = self.rbm_trainer.hidden_features(rbm_last)
        preds, probs = self.cnn_trainer.predict(cnn_last, rbm_feats)

        cls = int(preds[0])
        return {
            'symbol': symbol,
            'class': self.CLASS_NAMES[cls],
            'class_id': cls,
            'probs': {self.CLASS_NAMES[i]: float(p) for i, p in enumerate(probs[0])},
        }
