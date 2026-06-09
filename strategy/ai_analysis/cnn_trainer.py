"""
Trains the CNN from ai_modules/cnn on windowed stock features.

Labels
------
0 = expected short setup  (forward return < -threshold)
1 = flat                  (|forward return| <= threshold)
2 = expected long setup   (forward return >  threshold)
"""

import copy
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ai_modules.cnn.convolution_neural_network import ConvolutionNeuralNetwork

logger = logging.getLogger(__name__)


class CNNTrainer:
    def __init__(
        self,
        input_length: int,
        rbm_feature_dim: int = 0,
        num_classes: int = 3,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5,
        device: str | None = None,
    ):
        if ConvolutionNeuralNetwork._compute_flat_size(input_length) <= 0:
            raise ValueError(
                f'CNN input_length={input_length} is too small for two '
                f'(kernel=5, pool=2) conv/pool stages. Increase window_size or '
                f'include more features.'
            )
        self.input_length = input_length
        self.rbm_feature_dim = rbm_feature_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._has_rbm = rbm_feature_dim > 0

        self.model: ConvolutionNeuralNetwork | None = None

    # ---------------------------------------------------------------- train
    def train(
        self,
        cnn_x: np.ndarray,
        labels: np.ndarray,
        rbm_feats: np.ndarray | None = None,
        val_split: float = 0.2,
    ) -> None:
        """Train the CNN on windowed features with early stopping.

        Uses a chronological train/val split so validation = most recent samples.
        Tracks validation loss and restores the best model weights after training.
        """
        if cnn_x.shape[1] != self.input_length:
            raise ValueError(f'cnn_x second dim must be {self.input_length}; got {cnn_x.shape[1]}')
        if self._has_rbm:
            if rbm_feats is None:
                raise ValueError('rbm_feature_dim > 0 but rbm_feats is None')
            if rbm_feats.shape[1] != self.rbm_feature_dim:
                raise ValueError(f'rbm_feats second dim must be {self.rbm_feature_dim}; got {rbm_feats.shape[1]}')

        self.model = ConvolutionNeuralNetwork(
            input_length=self.input_length,
            num_classes=self.num_classes,
            rbm_features=self.rbm_feature_dim,
        ).to(self.device)

        x_img = torch.tensor(cnn_x, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(labels, dtype=torch.long)

        split = int(len(y) * (1.0 - val_split))

        if self._has_rbm:
            x_rbm = torch.tensor(rbm_feats, dtype=torch.float32)
            train_ds = TensorDataset(x_img[:split], x_rbm[:split], y[:split])
            val_ds = TensorDataset(x_img[split:], x_rbm[split:], y[split:])
        else:
            train_ds = TensorDataset(x_img[:split], y[:split])
            val_ds = TensorDataset(x_img[split:], y[split:])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch in train_loader:
                if self._has_rbm:
                    img, feat, target = batch
                    feat = feat.to(self.device)
                else:
                    img, target = batch
                    feat = None
                img = img.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                logits = self.model(img, feat)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * img.size(0)

            train_loss = running_loss / max(len(train_ds), 1)
            val_loss = self._evaluate_loss(val_loader, criterion) if len(val_ds) else float('nan')
            val_acc = self._evaluate(val_loader) if len(val_ds) else float('nan')

            logger.info(f'CNN epoch {epoch}/{self.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}')

            if len(val_ds) == 0:
                continue

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def _evaluate(self, loader: DataLoader) -> float:
        assert self.model is not None
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                if self._has_rbm:
                    img, feat, target = batch
                    feat = feat.to(self.device)
                else:
                    img, target = batch
                    feat = None
                img = img.to(self.device)
                target = target.to(self.device)
                logits = self.model(img, feat)
                preds = logits.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)
        return correct / max(total, 1)

    def _evaluate_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        assert self.model is not None
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in loader:
                if self._has_rbm:
                    img, feat, target = batch
                    feat = feat.to(self.device)
                else:
                    img, target = batch
                    feat = None
                img = img.to(self.device)
                target = target.to(self.device)
                logits = self.model(img, feat)
                loss = criterion(logits, target)
                total_loss += loss.item() * img.size(0)
                total_samples += img.size(0)
        return total_loss / max(total_samples, 1)

    # ------------------------------------------------------------- inference
    def predict(self, cnn_x: np.ndarray, rbm_feats: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        preds : (N,) int class labels
        probs : (N, num_classes) softmax probabilities
        """
        if self.model is None:
            raise RuntimeError('CNN has not been trained yet')
        self.model.eval()
        with torch.no_grad():
            img = torch.tensor(cnn_x, dtype=torch.float32).unsqueeze(1).to(self.device)
            feat = None
            if self._has_rbm and rbm_feats is not None:
                feat = torch.tensor(rbm_feats, dtype=torch.float32).to(self.device)
            logits = self.model(img, feat)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
        return preds, probs
