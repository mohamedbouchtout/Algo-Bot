"""
Wraps the legacy RBM (ai_modules.rbm.my_RBM_tf2_test.RBM) for stock-data training.

The RBM class itself is unchanged — we only supply the {'x_train', 'x_test'}
dict it expects, derived from binarised stock features, and expose a small
helper to extract hidden-layer activations as features for the CNN.
"""

import logging
import os
import sys
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# The RBM module relies on relative imports (`from datasets.bas_data import ...`
# inside small_Big mode) and lives at ai_modules/rbm/. Make sure that directory
# is on sys.path so those legacy imports keep working if someone flips small_Big.
_RBM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'ai_modules', 'rbm',
)
if _RBM_DIR not in sys.path:
    sys.path.insert(0, _RBM_DIR)

from ai_modules.rbm.my_RBM_tf2_test import RBM  # noqa: E402


class _SimpleOptimizer:
    """Matches the optimizer signature that RBM.train() expects."""

    def __init__(self, machine, lr: float = 0.05):
        self.machine = machine
        self.lr = lr

    def fit(self):
        m = self.machine
        m.weights.assign_add(self.lr * m.grad_dict['weights'])
        m.visible_biases.assign_add(self.lr * m.grad_dict['visible_biases'])
        m.hidden_biases.assign_add(self.lr * m.grad_dict['hidden_biases'])


class RBMTrainer:
    def __init__(
        self,
        visible_dim: int,
        hidden_dim: int = 64,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.05,
        k: int = 1,
        name: str = 'stock_rbm',
    ):
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.k = k
        self.name = name
        self.rbm: Optional[RBM] = None

    # -------------------------------------------------------------- training
    def train(self, x_train: np.ndarray, x_test: np.ndarray) -> None:
        if x_train.ndim != 2 or x_train.shape[1] != self.visible_dim:
            raise ValueError(
                f"x_train must be (N, {self.visible_dim}); got {x_train.shape}"
            )

        # RBM does integer sqrt of picture_shape in a few debug plots; give it
        # a sensible square-ish shape even if the vector length isn't a perfect
        # square — it's purely cosmetic when we don't plot.
        side = max(1, int(np.ceil(np.sqrt(self.visible_dim))))
        picture_shape = (side, side)

        self.rbm = RBM(
            visible_dim=self.visible_dim,
            hidden_dim=self.hidden_dim,
            number_of_epochs=self.epochs,
            picture_shape=picture_shape,
            batch_size=self.batch_size,
            initial_temperature=1,
            annealing_decay=0,
            training_algorithm='cd',
            k=self.k,
            n_test_samples=min(32, len(x_test)),
            NAME=self.name,
            initial_gamma=1.0,
            gamma_decay=0.0,
        )

        data = {
            'x_train': x_train.astype(np.float64),
            'x_test': x_test.astype(np.float64),
        }
        optimizer = _SimpleOptimizer(self.rbm, lr=self.learning_rate)
        logger.info(
            f"Training RBM: visible={self.visible_dim}, hidden={self.hidden_dim}, "
            f"epochs={self.epochs}, batch={self.batch_size}, samples={len(x_train)}"
        )
        self.rbm.train(data, optimizer)

    # ------------------------------------------------------------ inference
    def hidden_features(self, x: np.ndarray) -> np.ndarray:
        """
        Return hidden-layer probabilities for each input vector. These are the
        features the CNN concatenates onto its own conv output.
        """
        if self.rbm is None:
            raise RuntimeError("RBM has not been trained yet")
        import tensorflow as tf
        h_prob = tf.sigmoid(
            tf.tensordot(
                x.astype(np.float64),
                self.rbm.weights,
                axes=[[1], [1]],
            )
            + self.rbm.hidden_biases
        ).numpy()
        return h_prob.astype(np.float32)
