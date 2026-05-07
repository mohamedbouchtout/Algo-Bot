"""
The convolution neural network class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionNeuralNetwork(nn.Module):
    def __init__(self, input_length: int, num_classes: int = 3, rbm_features: int = 8):
        """
        Parameters
        ----------
        input_length : length of the 1-D input signal (e.g. window_size * n_features).
        num_classes  : number of output classes.
        rbm_features : size of the RBM hidden-feature vector that gets
                       concatenated before fc1.
        """
        super().__init__()
        self.input_length = input_length
        self.rbm_features = rbm_features

        self.conv1 = nn.Conv1d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout1d()

        # Compute the flattened size produced by the two (k=5, pool=2) stages
        flat_size = self._compute_flat_size(input_length)
        if flat_size <= 0:
            raise ValueError(
                f"input_length={input_length} is too small for two "
                f"(kernel=5, pool=2) conv/pool stages."
            )
        self._flat_size = flat_size

        self.fc1 = nn.Linear(flat_size + rbm_features, 64)
        self.fc2 = nn.Linear(64, num_classes)

    @staticmethod
    def _compute_flat_size(input_length: int) -> int:
        """Length after two conv(k=5) + max_pool(k=2) stages, times 16 channels."""
        L = input_length
        L = (L - 4) // 2   # conv1 (k=5, no padding) -> L-4, then max_pool1d(2)
        L = (L - 4) // 2   # conv2 (k=5, no padding) -> L-4, then max_pool1d(2)
        return 16 * max(L, 0)

    def forward(self, img, rbm_feats):
        x = F.max_pool1d(F.relu(self.conv1(img)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)                 # (batch, flat_size)
        x = torch.cat([x, rbm_feats], dim=1)      # (batch, flat_size + rbm_features)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
