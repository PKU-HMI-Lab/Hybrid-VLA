import os
import pathlib
import sys

import r3m
import torch
import torch.nn as nn
from vc_models.models.vit import model_utils


class R3MEncoder(nn.Module):
    def __init__(self, model_name, device, freeze=True):  # resnet18, resnet34, resnet50
        super(R3MEncoder, self).__init__()
        self.model = r3m.load_r3m(modelid=model_name).module.to(device)
        self.feature_dim = {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
        }[model_name]
        if freeze:
            self.freeze()

    def forward(self, x):
        # * https://github.com/facebookresearch/r3m/blob/main/r3m/example.py
        # * R3M expects image input to be [0-255]
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    import numpy as np

    from lift3d.helpers.common import Logger

    for model_name in ["resnet34"]:
        encoder = R3MEncoder(model_name=model_name, device="cuda:0").to("cuda:0")
        image = np.random.randint(0, 255, (3, 224, 224)).astype(np.float32)
        image = torch.tensor(image).unsqueeze(0).to("cuda:0")
        feat = encoder(image)
        for name, param in encoder.named_parameters():
            Logger.log_info(f"param {name}: {param.requires_grad}")
        Logger.log_info(f"feat shape: {feat.shape}")
        Logger.log_info(f"enbedding dim: {encoder.feature_dim}")
        Logger.print_seperator()
