"""
LSTM classifier for time-series stock feature windows.

Unlike the CNN which flattens the window into a 1-D signal, the LSTM
processes the window as a sequence of timesteps, each with n_features.
This preserves temporal ordering and lets the model learn dependencies
between consecutive days.
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))

        fc_input = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(fc_input, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (batch, window_size, n_features)
        lstm_out, _ = self.lstm(x)

        if self.bidirectional:
            last_out = torch.cat(
                [lstm_out[:, -1, : self.hidden_size], lstm_out[:, 0, self.hidden_size :]],
                dim=1,
            )
        else:
            last_out = lstm_out[:, -1, :]

        x = self.layer_norm(last_out)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
