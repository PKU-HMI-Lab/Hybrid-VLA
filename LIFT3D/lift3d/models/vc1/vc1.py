import os
import pathlib
import sys

import torch
import torch.nn as nn
import vc_models
from vc_models.models.vit import model_utils


class VC1Encoder(nn.Module):
    def __init__(self, model_name: str = "vc1_vitb", freeze=True):  # vc1_vitb, vc1_vitl
        super(VC1Encoder, self).__init__()
        model_key = (
            model_utils.VC1_BASE_NAME
            if model_name == "vc1_vitb"
            else model_utils.VC1_LARGE_NAME
        )
        self.model, self.feature_dim, self.preprocess, self.model_info = (
            model_utils.load_model(model_key)
        )
        if freeze:
            self.freeze()

    def forward(self, x):
        x = self.preprocess(x / 255.0)
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    import numpy as np

    from lift3d.helpers.common import Logger

    encoder = VC1Encoder().to("cuda:0")
    image = np.random.randint(0, 255, (3, 224, 224)).astype(np.float32)
    image = torch.tensor(image).unsqueeze(0).to("cuda:0")
    feat = encoder(image)
    for name, param in encoder.named_parameters():
        Logger.log_info(f"param {name}: {param.requires_grad}")
    Logger.log_info(f"feat shape: {feat.shape}")
    Logger.log_info(f"enbedding dim: {encoder.feature_dim}")
    Logger.log_info(f"model info: {encoder.model_info}")
