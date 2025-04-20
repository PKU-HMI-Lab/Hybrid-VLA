import os
import pathlib
import sys

import easydict
import torch
import torch.nn as nn
import yaml

sys.path.append(str(pathlib.Path(__file__).parent))
from openpoints.models.backbone import PointNextEncoder


class PointNextModel(nn.Module):
    def __init__(self, config_file):
        super(PointNextModel, self).__init__()
        if not os.path.isabs(config_file):
            parent_dir = pathlib.Path(__file__).parent.resolve()
            config_file = os.path.join(parent_dir, config_file)
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        config = easydict.EasyDict(config)
        self.model = PointNextEncoder(**config).float()
        self.feature_dim = 512

    def forward(self, point_clouds):
        point_clouds = point_clouds[:, :, :3]
        with torch.amp.autocast(device_type="cuda"):
            embedding = self.model.forward_cls_feat(point_clouds.contiguous())
            return embedding


if __name__ == "__main__":

    from helpers.common import Logger

    device = "cuda"
    point_clouds = torch.randn([4, 4096, 3]).to(device)
    # Test PointNext
    model = PointNextModel("point_next.yaml").to(device)
    embedding = model.forward(point_clouds)
    assert embedding.shape[1] == model.feature_dim
    Logger.log_info(f"feature dimension: {model.feature_dim}")
