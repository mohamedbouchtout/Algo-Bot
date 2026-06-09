"""
The convolution neural network class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_length: int,
        num_classes: int = 3,
        rbm_features: int = 0,
        dropout_rate: float = 0.3,
    ):
        """
        Parameters
        ----------
        input_length : length of the 1-D input signal (e.g. window_size * n_features).
        num_classes  : number of output classes.
        rbm_features : size of the RBM hidden-feature vector that gets
                       concatenated before fc1. Set to 0 to disable (default).
        dropout_rate : dropout probability applied after each conv/pool stage
                       and before the fully connected layers.
        """
        super().__init__()
        self.input_length = input_length
        self.rbm_features = rbm_features

        self.conv1 = nn.Conv1d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=5)
        self.conv_dropout = nn.Dropout1d(p=dropout_rate)
        self.dropout = nn.Dropout1d(p=dropout_rate)

        flat_size = self._compute_flat_size(input_length)
        if flat_size <= 0:
            raise ValueError(f'input_length={input_length} is too small for two (kernel=5, pool=2) conv/pool stages.')
        self._flat_size = flat_size

        self.fc1 = nn.Linear(flat_size + rbm_features, 64)
        self.fc2 = nn.Linear(64, num_classes)

    @staticmethod
    def _compute_flat_size(input_length: int) -> int:
        """Length after two conv(k=5) + max_pool(k=2) stages, times 16 channels."""
        L = input_length
        L = (L - 4) // 2
        L = (L - 4) // 2
        return 16 * max(L, 0)

    def forward(self, img, rbm_feats=None):
        x = self.conv_dropout(F.max_pool1d(F.relu(self.conv1(img)), 2))
        x = self.conv_dropout(F.max_pool1d(F.relu(self.conv2(x)), 2))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        if rbm_feats is not None and self.rbm_features > 0:
            x = torch.cat([x, rbm_feats], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
