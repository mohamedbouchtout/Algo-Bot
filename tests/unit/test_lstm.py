"""Unit tests for LSTM network and trainer."""

import numpy as np
import pytest
import torch

from ai_modules.lstm.lstm_network import LSTMClassifier
from strategy.ai_analysis.lstm_trainer import LSTMTrainer


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
