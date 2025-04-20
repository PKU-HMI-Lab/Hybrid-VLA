import os
import pathlib
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: List[int], output_dim: int, init_method: str
    ):
        super(MLP, self).__init__()

        # Model
        activation = nn.SELU()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        for li in range(len(hidden_dims)):
            if li == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[li], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[li], hidden_dims[li + 1]))
                layers.append(activation)
        self.model = nn.Sequential(*layers)

        # Initialize the model weights
        mlp_weights = [np.sqrt(2)] * len(hidden_dims)
        mlp_weights.append(0.01)
        assert init_method in ["orthogonal", "xavier_uniform"]
        self.init_weights(self.model, mlp_weights, init_method)

    @staticmethod
    def init_weights(sequential, scales, init_method):
        if init_method == "orthogonal":
            [
                torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
                for idx, module in enumerate(
                    mod for mod in sequential if isinstance(mod, nn.Linear)
                )
            ]
        elif init_method == "xavier_uniform":
            for module in sequential:
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs


if __name__ == "__main__":
    from lift3d.helpers.common import Logger

    mlp = MLP(
        input_dim=100,
        hidden_dims=[256, 128, 64],
        output_dim=7,
        init_method="orthogonal",
    )
    inputs = torch.randn(32, 100)
    outputs = mlp(inputs)
    for name, param in mlp.named_parameters():
        Logger.log_info(f"param {name}: {param.requires_grad}")
    Logger.log_info(f"Input shape: {inputs.shape}")
    Logger.log_info(f"Output shape: {outputs.shape}")
