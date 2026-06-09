"""
Trains the LSTM classifier on windowed stock features.

The LSTM receives features as a sequence of (window_size, n_features)
rather than the flattened vector the CNN uses. This preserves temporal
ordering within the window.

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

from ai_modules.lstm.lstm_network import LSTMClassifier

logger = logging.getLogger(__name__)


class LSTMTrainer:
    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5,
        dropout_rate: float = 0.3,
        bidirectional: bool = False,
        device: str | None = None,
    ):
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model: LSTMClassifier | None = None

    def _reshape_to_sequence(self, flat_x: np.ndarray) -> np.ndarray:
        """Reshape (N, window_size * n_features) -> (N, window_size, n_features)."""
        return flat_x.reshape(-1, self.window_size, self.n_features)

    # ---------------------------------------------------------------- train
    def train(
        self,
        cnn_x: np.ndarray,
        labels: np.ndarray,
        val_split: float = 0.2,
    ) -> None:
        """Train the LSTM on windowed features with early stopping.

        Accepts the same flattened input as CNNTrainer and reshapes it into
        a (batch, window_size, n_features) sequence internally. Uses gradient
        clipping (max_norm=1.0) to prevent exploding gradients.
        """
        seq_x = self._reshape_to_sequence(cnn_x)

        self.model = LSTMClassifier(
            n_features=self.n_features,
            window_size=self.window_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate,
            bidirectional=self.bidirectional,
        ).to(self.device)

        x_tensor = torch.tensor(seq_x, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)

        split = int(len(y_tensor) * (1.0 - val_split))
        train_ds = TensorDataset(x_tensor[:split], y_tensor[:split])
        val_ds = TensorDataset(x_tensor[split:], y_tensor[split:])

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
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item() * x_batch.size(0)

            train_loss = running_loss / max(len(train_ds), 1)
            val_loss = self._evaluate_loss(val_loader, criterion) if len(val_ds) else float('nan')
            val_acc = self._evaluate(val_loader) if len(val_ds) else float('nan')

            logger.info(f'LSTM epoch {epoch}/{self.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}')

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
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.model(x_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        return correct / max(total, 1)

    def _evaluate_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        assert self.model is not None
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * x_batch.size(0)
                total_samples += x_batch.size(0)
        return total_loss / max(total_samples, 1)

    # ------------------------------------------------------------- inference
    def predict(self, cnn_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        preds : (N,) int class labels
        probs : (N, num_classes) softmax probabilities
        """
        if self.model is None:
            raise RuntimeError('LSTM has not been trained yet')
        self.model.eval()
        seq_x = self._reshape_to_sequence(cnn_x)
        with torch.no_grad():
            x_tensor = torch.tensor(seq_x, dtype=torch.float32).to(self.device)
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
        return preds, probs
