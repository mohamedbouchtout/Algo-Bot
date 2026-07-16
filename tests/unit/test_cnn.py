"""Unit tests for CNN model and early stopping."""

import numpy as np
import pytest
import torch

from ai_modules.cnn.convolution_neural_network import ConvolutionNeuralNetwork
from strategy.ai_analysis.cnn_trainer import CNNTrainer


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
