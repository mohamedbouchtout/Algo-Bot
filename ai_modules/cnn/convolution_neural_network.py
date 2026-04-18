"""
The convolution neural network class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionNeuralNetwork(nn.Module):
    def __init__(self, num_classes=3, rbm_features=8):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout1d()
        self.fc1 = nn.Linear(256 + rbm_features, 64)  # concatenate RBM features here
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, img, rbm_feats):
        x = F.max_pool1d(F.relu(self.conv1(img)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)          # (batch, 256)
        x = torch.cat([x, rbm_feats], dim=1)  # (batch, 264)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
