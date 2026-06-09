"""
Walk-forward (expanding window) cross-validation for time series.

Unlike k-fold CV, walk-forward validation respects temporal ordering:
the training set always precedes the validation set, and the training
window expands with each fold.
"""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    def __init__(
        self,
        n_splits: int = 5,
        min_train_pct: float = 0.5,
    ):
        if n_splits < 2:
            raise ValueError('n_splits must be >= 2')
        if not 0.1 <= min_train_pct < 1.0:
            raise ValueError('min_train_pct must be in [0.1, 1.0)')
        self.n_splits = n_splits
        self.min_train_pct = min_train_pct

    def split(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate expanding-window train/val index splits.

        The first training window covers min_train_pct of the data.
        The remaining data is divided into n_splits equal-sized validation
        blocks. Each fold trains on everything before its validation block.
        """
        initial_train_size = int(n_samples * self.min_train_pct)
        remaining = n_samples - initial_train_size
        step_size = max(1, remaining // self.n_splits)

        splits = []
        for i in range(self.n_splits):
            val_start = initial_train_size + i * step_size
            val_end = min(val_start + step_size, n_samples)

            if val_start >= n_samples or val_end <= val_start:
                break

            train_idx = np.arange(0, val_start)
            val_idx = np.arange(val_start, val_end)
            splits.append((train_idx, val_idx))

        logger.info(f'Walk-forward: {len(splits)} folds from {n_samples} samples (initial train={initial_train_size})')
        return splits
