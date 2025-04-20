from typing import List

import torch
import torch.nn as nn


class BatchNormMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        nonlinearity: str = "relu",  # either 'tanh' or 'relu'
        dropout_rate: float = 0.0,
    ):
        super(BatchNormMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_sizes = (
            [
                input_dim,
            ]
            + hidden_dims
            + [
                output_dim,
            ]
        )
        self.fc_layers = nn.ModuleList(
            [
                nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1])
                for i in range(len(self.layer_sizes) - 1)
            ]
        )
        self.nonlinearity = torch.relu if nonlinearity == "relu" else torch.tanh
        self.input_batchnorm = nn.BatchNorm1d(num_features=input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = x
        out = self.input_batchnorm(out)
        for i in range(len(self.fc_layers) - 1):
            out = self.fc_layers[i](out)
            out = self.dropout(out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        return out
