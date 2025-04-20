import os
import pathlib
import sys

import spa
import spa.models
import torch
import torch.nn as nn


class SPAEncoder(nn.Module):
    def __init__(self, model_name, freeze=True):  # vit_base, vit_large
        super(SPAEncoder, self).__init__()
        self.model = (
            spa.models.spa_vit_base_patch16(pretrained=True)
            if model_name == "vit_base"
            else spa.models.spa_vit_large_patch16(pretrained=True)
        )
        self.feature_dim = {
            "vit_base": 768,
            "vit_large": 1024,
        }[model_name]
        if freeze:
            self.freeze()

    def forward(self, x):
        # image transform here:
        # https://github.com/HaoyiZhu/SPA/blob/main/spa/models/components/img_backbones/vit.py#L69
        return self.model(x / 255.0)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    import numpy as np

    from lift3d.helpers.common import Logger

    for model_name in ["vit_base", "vit_large"]:
        encoder = SPAEncoder(model_name=model_name).to("cuda:0")
        image = np.random.randint(0, 255, (3, 224, 224)).astype(np.float32)
        image = torch.tensor(image).unsqueeze(0).to("cuda:0")
        feat = encoder(image)
        for name, param in encoder.named_parameters():
            Logger.log_info(f"param {name}: {param.requires_grad}")
        Logger.log_info(f"feat shape: {feat.shape}")
        # Logger.log_info(f'enbedding dim: {encoder.feature_dim}')
        Logger.print_seperator()
