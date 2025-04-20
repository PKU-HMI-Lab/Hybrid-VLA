import os
import pathlib
import sys

import clip
import torch
import torch.nn as nn
from torchvision.transforms import Normalize


class CLIPEncoder(nn.Module):
    def __init__(self, model_name, freeze=True):
        super(CLIPEncoder, self).__init__()
        self.model, self.preprocess = clip.load(model_name)
        # see: https://github.com/openai/CLIP/blob/main/clip/clip.py line 79
        self.preprocess = Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        self.feature_dim = {
            "ViT-B/32": 512,
        }[model_name]
        if freeze:
            self.freeze()

    def forward(self, x):
        x = self.preprocess(x)
        return self.model.encode_image(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    import numpy as np

    from lift3d.helpers.common import Logger

    encoder = CLIPEncoder("ViT-B/32").to("cuda:0")
    image = np.random.randint(0, 255, (3, 224, 224)).astype(np.float32)
    image = torch.tensor(image).unsqueeze(0).to("cuda:0")
    feat = encoder(image)
    for name, param in encoder.named_parameters():
        Logger.log_info(f"param {name}: {param.requires_grad}")
    Logger.log_info(f"feat shape: {feat.shape}")
    Logger.log_info(f"enbedding dim: {encoder.feature_dim}")
