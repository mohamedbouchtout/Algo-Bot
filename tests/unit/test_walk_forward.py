"""Unit tests for walk-forward cross-validation splits."""

import pytest

from strategy.ai_analysis.walk_forward import WalkForwardValidator


class TestWalkForwardSplits:
    def test_correct_number_of_folds(self):
        validator = WalkForwardValidator(n_splits=5, min_train_pct=0.5)
        splits = validator.split(1000)
        assert len(splits) == 5

    def test_no_data_leakage(self):
        validator = WalkForwardValidator(n_splits=5, min_train_pct=0.5)
        splits = validator.split(1000)
        for train_idx, val_idx in splits:
            assert train_idx.max() < val_idx.min()

    def test_expanding_window(self):
        validator = WalkForwardValidator(n_splits=5, min_train_pct=0.5)
        splits = validator.split(1000)
        train_sizes = [len(t) for t, _ in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_complete_coverage(self):
        validator = WalkForwardValidator(n_splits=5, min_train_pct=0.5)
        splits = validator.split(1000)
        all_indices = set()
        for train_idx, val_idx in splits:
            all_indices.update(train_idx.tolist())
            all_indices.update(val_idx.tolist())
        assert len(all_indices) == 1000

    def test_minimum_fold_sizes(self):
        validator = WalkForwardValidator(n_splits=3, min_train_pct=0.5)
        splits = validator.split(100)
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0

    def test_invalid_n_splits(self):
        with pytest.raises(ValueError):
            WalkForwardValidator(n_splits=1)
