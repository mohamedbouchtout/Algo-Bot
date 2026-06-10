"""
High-level orchestrator for the AI analysis pipeline.

Responsibilities
----------------
* Pull historical data for a list of tickers via the existing StockDataFetcher.
* Build a combined training set with FeatureBuilder.
* Train the CNN on continuous windowed features -> forward-return label.
* Expose `predict(symbol)` that pulls the latest bars and returns a LONG/FLAT/
  SHORT classification plus class probabilities.
* Optionally run walk-forward cross-validation to evaluate model quality.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_fetch.historical_data import StockDataFetcher
from execution.risk_manager import RiskManager
from strategy.ai_analysis.cnn_trainer import CNNTrainer
from strategy.ai_analysis.data_preparation.feature_builder import FeatureBuilder
from strategy.ai_analysis.lstm_trainer import LSTMTrainer
from strategy.ai_analysis.walk_forward import WalkForwardValidator

logger = logging.getLogger(__name__)


class AIAnalyzer:
    CLASS_NAMES = {0: 'SHORT', 1: 'FLAT', 2: 'LONG'}

    VALID_MODEL_TYPES = ('cnn', 'lstm')

    def __init__(
        self,
        stock_data: StockDataFetcher,
        feature_builder: FeatureBuilder | None = None,
        cnn_epochs: int = 100,
        params: dict | None = None,
        model_type: str = 'lstm',
        # Deprecated — kept for backward compat, ignored
        rbm_hidden_dim: int = 64,
        rbm_epochs: int = 30,
    ):
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {self.VALID_MODEL_TYPES}, got '{model_type}'")
        self.stock_data = stock_data
        self.feature_builder = feature_builder or FeatureBuilder(window_size=10, n_bits=4)
        self.model_type = model_type
        self._trainer = None
        self.cnn_epochs = cnn_epochs
        self.params = params or {}

        self._bar_cache: dict[str, pd.DataFrame] = {}
        self._continuous_per_ticker: list[pd.DataFrame] = []
        self._kept_tickers: list[str] = []

    def _create_trainer(self):
        """Create the appropriate trainer (LSTM or CNN) based on model_type."""
        fb = self.feature_builder
        if self.model_type == 'lstm':
            return LSTMTrainer(
                n_features=len(fb.feature_names),
                window_size=fb.window_size,
                epochs=self.cnn_epochs,
            )
        return CNNTrainer(
            input_length=fb.cnn_input_length,
            epochs=self.cnn_epochs,
        )

    def _get_bars(self, symbol: str) -> pd.DataFrame | None:
        if symbol in self._bar_cache:
            return self._bar_cache[symbol]
        df = self.stock_data.get_historical_data(symbol, self.params['ai_analyzer']['lookback_days'])
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

        Returns True if the ticker was added, False if skipped.
        """
        if symbol in self._kept_tickers:
            logger.debug(f'{symbol}: already in dataset, skipping')
            return False

        bars = self._get_bars(symbol)
        if bars is None or len(bars) < 250:
            logger.info(f'Skipping {symbol}: insufficient bars')
            return False

        try:
            feats = self.feature_builder.build_continuous_features(bars)
        except ValueError as e:
            logger.warning(f'{symbol}: {e}')
            return False

        self._continuous_per_ticker.append(feats)
        self._kept_tickers.append(symbol)
        logger.debug(f'Added {symbol} to dataset ({len(self._kept_tickers)} tickers accumulated)')
        return True

    def build_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assemble the pooled training tensors.

        Returns
        -------
        cnn_x  : (N, input_length) float32
        labels : (N,) int64
        ids    : (N,) int64 per-sample ticker index
        """
        if not self._continuous_per_ticker:
            raise RuntimeError('No tickers in dataset, call add_ticker() (or train(tickers)) first')

        self.feature_builder.fit_bin_edges(self._continuous_per_ticker)

        cnn_chunks, label_chunks, ticker_ids = [], [], []
        for idx, sym in enumerate(self._kept_tickers):
            bars = self._bar_cache[sym]
            _, cnn_x, labels = self.feature_builder.build_windows(bars, include_labels=True, include_rbm=False)
            if len(cnn_x) == 0:
                continue
            cnn_chunks.append(cnn_x)
            label_chunks.append(labels)
            ticker_ids.append(np.full(len(cnn_x), idx, dtype=np.int64))

        if not cnn_chunks:
            raise RuntimeError('No tickers produced usable windowed samples')

        cnn_all = np.concatenate(cnn_chunks, axis=0)
        labels_all = np.concatenate(label_chunks, axis=0)
        ids_all = np.concatenate(ticker_ids, axis=0)

        logger.info(
            f'Dataset built: {len(cnn_all)} samples across {len(self._kept_tickers)} tickers '
            f'(cnn_len={cnn_all.shape[1]}, '
            f'class counts={np.bincount(labels_all, minlength=3).tolist()})'
        )
        return cnn_all, labels_all, ids_all

    def finalize_training(self, val_split: float = 0.2) -> None:
        """
        Fit the model on everything accumulated via `add_ticker()`.
        Training on a single ticker is technically allowed but strongly
        discouraged (models will just memorise that ticker).
        """
        if len(self._kept_tickers) < 2:
            logger.warning(
                f'finalize_training() called with only {len(self._kept_tickers)} ticker(s); pooled training needs several tickers to generalise.'
            )

        cnn_x, labels, _ = self.build_dataset()

        self._trainer = self._create_trainer()
        self._trainer.train(cnn_x, labels, val_split=val_split)

    def train(self, tickers: list[str], val_split: float = 0.2) -> None:
        """
        Convenience wrapper: accumulate every ticker in `tickers`, then fit.
        Equivalent to calling `add_ticker()` in a loop and then
        `finalize_training()`.
        """
        for sym in tickers:
            self.add_ticker(sym)
        self.finalize_training(val_split=val_split)

    def walk_forward_train(self, n_splits: int = 5) -> dict:
        """
        Run walk-forward cross-validation and train a final model.

        Returns per-fold metrics and averages. The final model (trained on
        all data) is stored in self._trainer so predict() works.
        """
        cnn_x, labels, _ = self.build_dataset()
        validator = WalkForwardValidator(n_splits=n_splits)
        splits = validator.split(len(cnn_x))

        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            trainer = self._create_trainer()
            train_x, train_y = cnn_x[train_idx], labels[train_idx]
            val_x, val_y = cnn_x[val_idx], labels[val_idx]

            trainer.train(train_x, train_y, val_split=0.0)

            preds, probs = trainer.predict(val_x)
            accuracy = (preds == val_y).mean()
            fold_metrics.append(
                {
                    'fold': fold_idx,
                    'train_size': len(train_idx),
                    'val_size': len(val_idx),
                    'accuracy': float(accuracy),
                }
            )
            logger.info(f'Walk-forward fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}, acc={accuracy:.3f}')

        avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
        logger.info(f'Walk-forward avg accuracy: {avg_acc:.3f}')

        self._trainer = self._create_trainer()
        self._trainer.train(cnn_x, labels, val_split=0.1)

        return {
            'folds': fold_metrics,
            'avg_accuracy': float(avg_acc),
            'total_samples': len(cnn_x),
        }

    def predict(self, symbol: str) -> dict | None:
        """
        Fetch the latest bars for `symbol`, build the most recent window and
        classify it. Returns None if anything is missing / not trained.
        """
        if self._trainer is None:
            raise RuntimeError('Call train() or finalize_training() before predict()')

        df = self.stock_data.get_historical_data(symbol, self.params['ai_analyzer']['lookback_days'])
        if df is None or len(df) < 250:
            return None

        _, cnn_x, _ = self.feature_builder.build_windows(df, include_labels=False, include_rbm=False)
        if len(cnn_x) == 0:
            return None

        cnn_last = cnn_x[-1:]
        preds, probs = self._trainer.predict(cnn_last)

        cls = int(preds[0])
        return {
            'symbol': symbol,
            'class': self.CLASS_NAMES[cls],
            'class_id': cls,
            'probs': {self.CLASS_NAMES[i]: float(p) for i, p in enumerate(probs[0])},
        }

    def construct_signal(self, df: pd.DataFrame, params, class_type: str, confidence: float) -> dict | None:
        """Construct a trading signal dict based on the most recent prediction."""
        entry_price = df['close'].iloc[-1]
        symbol = df['symbol'].iloc[0]
        target_price = 0
        stop_loss = 0
        risk = 0

        risk_manager = RiskManager(params)
        sl = risk_manager.get_stop_loss_pct(df)

        if class_type == 'LONG':
            stop_loss = df['close'].iloc[-1] * (1 - sl)
            risk = entry_price - stop_loss
            target_price = entry_price + (risk * params['ai_analyzer']['risk_reward_ratio'])
        elif class_type == 'SHORT':
            stop_loss = df['close'].iloc[-1] * (1 + sl)
            risk = stop_loss - entry_price
            target_price = entry_price - (risk * params['ai_analyzer']['risk_reward_ratio'])

        if entry_price < 1:
            entry_price = round(entry_price, 4)
            target_price = round(target_price, 4)
            stop_loss = round(stop_loss, 4)
        else:
            entry_price = round(entry_price, 2)
            target_price = round(target_price, 2)
            stop_loss = round(stop_loss, 2)

        return {
            'strategy_type': 'ai_analysis',
            'type': class_type,
            'symbol': symbol,
            'entry': entry_price,
            'stop': stop_loss,
            'target': target_price,
            'risk': risk,
            'reward': risk * params['ai_analyzer']['risk_reward_ratio'],
            'confidence': confidence,
        }
