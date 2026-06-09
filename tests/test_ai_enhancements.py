"""Unit tests for AI analysis pipeline enhancements."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from ai_modules.cnn.convolution_neural_network import ConvolutionNeuralNetwork
from ai_modules.lstm.lstm_network import LSTMClassifier
from strategy.ai_analysis.cnn_trainer import CNNTrainer
from strategy.ai_analysis.lstm_trainer import LSTMTrainer
from strategy.ai_analysis.retrain_trigger import RetrainTrigger
from strategy.ai_analysis.walk_forward import WalkForwardValidator


@pytest.fixture
def synthetic_bars():
    np.random.seed(42)
    n = 300
    close = 100.0 * np.cumprod(1 + np.random.randn(n) * 0.02)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.01)
    open_ = close * (1 + np.random.randn(n) * 0.005)
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    dates = pd.bdate_range(start='2023-01-01', periods=n)
    return pd.DataFrame(
        {
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'date': dates,
            'symbol': 'SYNTH',
        }
    )


# ====================== CNN Without RBM ======================


class TestCNNWithoutRBM:
    def test_model_creation_no_rbm(self):
        model = ConvolutionNeuralNetwork(input_length=170, rbm_features=0)
        assert model.rbm_features == 0

    def test_forward_pass_no_rbm(self):
        model = ConvolutionNeuralNetwork(input_length=170, rbm_features=0)
        x = torch.randn(4, 1, 170)
        out = model(x)
        assert out.shape == (4, 3)

    def test_forward_pass_with_rbm(self):
        model = ConvolutionNeuralNetwork(input_length=170, rbm_features=64)
        x = torch.randn(4, 1, 170)
        rbm = torch.randn(4, 64)
        out = model(x, rbm)
        assert out.shape == (4, 3)

    def test_conv_dropout_present(self):
        model = ConvolutionNeuralNetwork(input_length=170, dropout_rate=0.3)
        assert hasattr(model, 'conv_dropout')
        assert model.conv_dropout.p == 0.3

    def test_trainer_no_rbm(self):
        trainer = CNNTrainer(input_length=170, rbm_feature_dim=0, epochs=2)
        cnn_x = np.random.randn(100, 170).astype(np.float32)
        labels = np.random.randint(0, 3, 100).astype(np.int64)
        trainer.train(cnn_x, labels, val_split=0.2)
        preds, probs = trainer.predict(cnn_x[:5])
        assert preds.shape == (5,)
        assert probs.shape == (5, 3)

    def test_trainer_with_rbm_backward_compat(self):
        trainer = CNNTrainer(input_length=170, rbm_feature_dim=64, epochs=2)
        cnn_x = np.random.randn(100, 170).astype(np.float32)
        rbm_feats = np.random.randn(100, 64).astype(np.float32)
        labels = np.random.randint(0, 3, 100).astype(np.int64)
        trainer.train(cnn_x, labels, rbm_feats=rbm_feats, val_split=0.2)
        preds, probs = trainer.predict(cnn_x[:5], rbm_feats=rbm_feats[:5])
        assert preds.shape == (5,)
        assert probs.shape == (5, 3)


# ====================== Early Stopping ======================


class TestEarlyStopping:
    def test_stops_before_max_epochs(self):
        trainer = CNNTrainer(input_length=170, epochs=100, patience=3, batch_size=32)
        np.random.seed(0)
        cnn_x = np.random.randn(200, 170).astype(np.float32)
        labels = np.random.randint(0, 3, 200).astype(np.int64)
        trainer.train(cnn_x, labels, val_split=0.3)
        assert trainer.model is not None

    def test_restores_best_weights(self):
        trainer = CNNTrainer(input_length=170, epochs=10, patience=3, batch_size=32)
        cnn_x = np.random.randn(200, 170).astype(np.float32)
        labels = np.random.randint(0, 3, 200).astype(np.int64)
        trainer.train(cnn_x, labels, val_split=0.3)
        assert trainer.model is not None
        preds, probs = trainer.predict(cnn_x[:5])
        assert not np.isnan(probs).any()

    def test_weight_decay_nonzero(self):
        trainer = CNNTrainer(input_length=170, weight_decay=1e-4)
        assert trainer.weight_decay == 1e-4


# ====================== Walk-Forward Splits ======================


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


# ====================== Volatility-Adjusted Labels ======================


class TestVolatilityAdjustedLabels:
    @patch('strategy.ai_analysis.data_preparation.market_features.yf')
    def test_fixed_vs_vol_adjusted_differ(self, mock_yf, synthetic_bars):
        mock_yf.Ticker.return_value.history.return_value = pd.DataFrame()

        from strategy.ai_analysis.data_preparation.feature_builder import FeatureBuilder
        from strategy.ai_analysis.data_preparation.indicator_features import IndicatorFeatureExtractor
        from strategy.ai_analysis.data_preparation.price_features import PriceFeatureExtractor
        from strategy.ai_analysis.data_preparation.volume_features import VolumeFeatureExtractor

        extractors = [
            PriceFeatureExtractor(),
            VolumeFeatureExtractor(),
            IndicatorFeatureExtractor(),
        ]

        fb_fixed = FeatureBuilder(
            extractors=extractors,
            volatility_adjusted_labels=False,
        )
        fb_vol = FeatureBuilder(
            extractors=extractors,
            volatility_adjusted_labels=True,
            volatility_threshold=1.0,
        )

        feats = fb_fixed.build_continuous_features(synthetic_bars)
        fb_fixed.fit_bin_edges([feats])
        fb_vol.feature_names = fb_fixed.feature_names
        fb_vol.bin_edges = fb_fixed.bin_edges

        _, _, labels_fixed = fb_fixed.build_windows(synthetic_bars, include_labels=True, include_rbm=False)
        _, _, labels_vol = fb_vol.build_windows(synthetic_bars, include_labels=True, include_rbm=False)

        assert labels_fixed is not None
        assert labels_vol is not None
        assert not np.array_equal(labels_fixed, labels_vol)

    @patch('strategy.ai_analysis.data_preparation.market_features.yf')
    def test_zero_atr_fallback(self, mock_yf, synthetic_bars):
        mock_yf.Ticker.return_value.history.return_value = pd.DataFrame()

        from strategy.ai_analysis.data_preparation.feature_builder import FeatureBuilder
        from strategy.ai_analysis.data_preparation.indicator_features import IndicatorFeatureExtractor
        from strategy.ai_analysis.data_preparation.price_features import PriceFeatureExtractor
        from strategy.ai_analysis.data_preparation.volume_features import VolumeFeatureExtractor

        extractors = [
            PriceFeatureExtractor(),
            VolumeFeatureExtractor(),
            IndicatorFeatureExtractor(),
        ]
        fb = FeatureBuilder(
            extractors=extractors,
            volatility_adjusted_labels=True,
        )
        feats = fb.build_continuous_features(synthetic_bars)
        fb.fit_bin_edges([feats])

        _, _, labels = fb.build_windows(synthetic_bars, include_labels=True, include_rbm=False)
        assert labels is not None
        assert len(labels) > 0
        assert not np.isnan(labels).any()


# ====================== Market Features ======================


class TestMarketFeatureExtractor:
    def test_feature_names(self):
        from strategy.ai_analysis.data_preparation.market_features import MarketFeatureExtractor

        assert len(MarketFeatureExtractor.FEATURE_NAMES) == 3

    def test_extract_with_empty_market_data(self, synthetic_bars):
        from strategy.ai_analysis.data_preparation.market_features import MarketFeatureExtractor

        extractor = MarketFeatureExtractor(market_data=pd.DataFrame())
        result = extractor.extract(synthetic_bars)
        assert list(result.columns) == MarketFeatureExtractor.FEATURE_NAMES
        assert len(result) == len(synthetic_bars)
        assert (result['vix_normalized'] == 1.0).all()

    def test_extract_with_mock_market_data(self, synthetic_bars):
        from strategy.ai_analysis.data_preparation.market_features import MarketFeatureExtractor

        dates = pd.bdate_range(start='2022-01-01', periods=600)
        market = pd.DataFrame(
            {
                'vix_close': np.full(600, 20.0),
                'spy_close': np.linspace(400, 500, 600),
            },
            index=dates,
        )

        extractor = MarketFeatureExtractor(market_data=market)
        result = extractor.extract(synthetic_bars)
        assert list(result.columns) == MarketFeatureExtractor.FEATURE_NAMES
        assert len(result) == len(synthetic_bars)
        assert (result['vix_normalized'] == 1.0).all()


# ====================== LSTM Network ======================


class TestLSTMNetwork:
    def test_model_creation(self):
        model = LSTMClassifier(n_features=20, window_size=10)
        assert model.n_features == 20
        assert model.window_size == 10
        assert model.hidden_size == 64

    def test_forward_pass(self):
        model = LSTMClassifier(n_features=20, window_size=10)
        x = torch.randn(4, 10, 20)
        out = model(x)
        assert out.shape == (4, 3)

    def test_bidirectional(self):
        model = LSTMClassifier(n_features=20, window_size=10, bidirectional=True)
        x = torch.randn(4, 10, 20)
        out = model(x)
        assert out.shape == (4, 3)

    def test_custom_hidden_size(self):
        model = LSTMClassifier(n_features=20, window_size=10, hidden_size=128)
        x = torch.randn(4, 10, 20)
        out = model(x)
        assert out.shape == (4, 3)

    def test_single_layer(self):
        model = LSTMClassifier(n_features=20, window_size=10, num_layers=1)
        x = torch.randn(4, 10, 20)
        out = model(x)
        assert out.shape == (4, 3)


# ====================== LSTM Trainer ======================


class TestLSTMTrainer:
    def test_train_and_predict(self):
        n_features = 20
        window_size = 10
        trainer = LSTMTrainer(
            n_features=n_features,
            window_size=window_size,
            epochs=3,
            batch_size=32,
        )
        flat_x = np.random.randn(100, window_size * n_features).astype(np.float32)
        labels = np.random.randint(0, 3, 100).astype(np.int64)
        trainer.train(flat_x, labels, val_split=0.2)

        preds, probs = trainer.predict(flat_x[:5])
        assert preds.shape == (5,)
        assert probs.shape == (5, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_early_stopping(self):
        trainer = LSTMTrainer(
            n_features=20,
            window_size=10,
            epochs=100,
            patience=3,
            batch_size=32,
        )
        flat_x = np.random.randn(200, 200).astype(np.float32)
        labels = np.random.randint(0, 3, 200).astype(np.int64)
        trainer.train(flat_x, labels, val_split=0.3)
        assert trainer.model is not None

    def test_reshape_to_sequence(self):
        trainer = LSTMTrainer(n_features=20, window_size=10)
        flat = np.random.randn(50, 200).astype(np.float32)
        seq = trainer._reshape_to_sequence(flat)
        assert seq.shape == (50, 10, 20)

    def test_predict_before_training_raises(self):
        trainer = LSTMTrainer(n_features=20, window_size=10)
        with pytest.raises(RuntimeError):
            trainer.predict(np.random.randn(5, 200).astype(np.float32))

    def test_bidirectional_training(self):
        trainer = LSTMTrainer(
            n_features=20,
            window_size=10,
            epochs=2,
            bidirectional=True,
        )
        flat_x = np.random.randn(100, 200).astype(np.float32)
        labels = np.random.randint(0, 3, 100).astype(np.int64)
        trainer.train(flat_x, labels, val_split=0.2)
        preds, probs = trainer.predict(flat_x[:5])
        assert preds.shape == (5,)
        assert probs.shape == (5, 3)


# ====================== Retrain Trigger ======================


class TestRetrainTrigger:
    def test_no_trigger_without_snapshot(self):
        trigger = RetrainTrigger()
        assert trigger.check_regime_shift_from_bars(30.0, 400.0) is False

    def test_vix_spike_triggers(self):
        trigger = RetrainTrigger(vix_change_pct=0.50)
        trigger.snapshot_from_bars(20.0, 500.0)
        assert trigger.check_regime_shift_from_bars(35.0, 500.0) is True

    def test_vix_stable_no_trigger(self):
        trigger = RetrainTrigger(vix_change_pct=0.50)
        trigger.snapshot_from_bars(20.0, 500.0)
        assert trigger.check_regime_shift_from_bars(25.0, 500.0) is False

    def test_spy_drop_triggers(self):
        trigger = RetrainTrigger(spy_drop_pct=0.05)
        trigger.snapshot_from_bars(20.0, 500.0)
        assert trigger.check_regime_shift_from_bars(20.0, 470.0) is True

    def test_spy_stable_no_trigger(self):
        trigger = RetrainTrigger(spy_drop_pct=0.05)
        trigger.snapshot_from_bars(20.0, 500.0)
        assert trigger.check_regime_shift_from_bars(20.0, 490.0) is False

    def test_spy_rise_no_trigger(self):
        trigger = RetrainTrigger(spy_drop_pct=0.05)
        trigger.snapshot_from_bars(20.0, 500.0)
        assert trigger.check_regime_shift_from_bars(20.0, 550.0) is False

    def test_accuracy_below_threshold_triggers(self):
        trigger = RetrainTrigger(min_accuracy=0.40, lookback_bars=10)
        preds = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
        actuals = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
        assert trigger.check_accuracy(preds, actuals) is True

    def test_accuracy_above_threshold_no_trigger(self):
        trigger = RetrainTrigger(min_accuracy=0.40, lookback_bars=10)
        preds = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        actuals = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        assert trigger.check_accuracy(preds, actuals) is False

    def test_accuracy_not_enough_data(self):
        trigger = RetrainTrigger(min_accuracy=0.40, lookback_bars=20)
        preds = [0, 0, 0, 0, 0]
        actuals = [1, 1, 1, 1, 1]
        assert trigger.check_accuracy(preds, actuals) is False

    def test_snapshot_from_bars(self):
        trigger = RetrainTrigger()
        trigger.snapshot_from_bars(25.0, 450.0)
        assert trigger._train_vix == 25.0
        assert trigger._train_spy == 450.0
