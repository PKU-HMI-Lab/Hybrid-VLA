import abc
from typing import List

import clip
import numpy as np
import torch
import torch.nn as nn

from lift3d.helpers.graphics import PointCloud
from lift3d.models.mlp.batchnorm_mlp import BatchNormMLP
from lift3d.models.mlp.mlp import MLP


class Actor(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, images, point_clouds, robot_states):
        pass


class VisionGuidedMLP(Actor):
    def __init__(
        self,
        image_encoder: nn.Module,
        image_dropout_rate: float,
        robot_state_dim: int,
        robot_state_dropout_rate: float,
        action_dim: int,
        policy_hidden_dims: List[int],
        policy_head_init_method: str,
    ):
        super(VisionGuidedMLP, self).__init__()
        self.image_encoder = image_encoder
        self.image_dropout = nn.Dropout(image_dropout_rate)
        self.robot_state_encoder = nn.Linear(robot_state_dim, image_encoder.feature_dim)
        self.robot_state_dropout = nn.Dropout(robot_state_dropout_rate)
        self.policy_head = MLP(
            input_dim=2 * image_encoder.feature_dim,
            hidden_dims=policy_hidden_dims,
            output_dim=action_dim,
            init_method=policy_head_init_method,
        )

    def forward(self, images, point_clouds, robot_states, texts):
        image_emb = self.image_encoder(images)
        image_emb = self.image_dropout(image_emb)
        robot_state_emb = self.robot_state_encoder(robot_states)
        robot_state_emb = self.robot_state_dropout(robot_state_emb)
        emb = torch.cat([image_emb, robot_state_emb], dim=1)
        actions = self.policy_head(emb)
        return actions


class PointCloudGuidedMLP(Actor):

    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        point_cloud_dropout_rate: float,
        robot_state_dim: int,
        robot_state_dropout_rate: float,
        action_dim: int,
        policy_hidden_dims: List[int],
        policy_head_init_method: str,
    ):
        super(PointCloudGuidedMLP, self).__init__()
        self.point_cloud_encoder = point_cloud_encoder
        self.point_cloud_dropout = nn.Dropout(point_cloud_dropout_rate)
        self.robot_state_encoder = nn.Linear(
            robot_state_dim, point_cloud_encoder.feature_dim
        )
        self.robot_state_dropout = nn.Dropout(robot_state_dropout_rate)
        self.policy_head = MLP(
            input_dim=2 * point_cloud_encoder.feature_dim,
            hidden_dims=policy_hidden_dims,
            output_dim=action_dim,
            init_method=policy_head_init_method,
        )

    def forward(self, images, point_clouds, robot_states, texts):
        # * Notice: normalize the input point cloud
        point_clouds = PointCloud.normalize(point_clouds)
        point_cloud_emb = self.point_cloud_encoder(point_clouds)
        point_cloud_emb = self.point_cloud_dropout(point_cloud_emb)
        robot_state_emb = self.robot_state_encoder(robot_states)
        robot_state_emb = self.robot_state_dropout(robot_state_emb)
        emb = torch.cat([point_cloud_emb, robot_state_emb], dim=1)
        actions = self.policy_head(emb)
        return actions


class VisionGuidedBatchNormMLP(Actor):
    def __init__(
        self,
        image_encoder: nn.Module,
        robot_state_dim: int,
        action_dim: int,
        policy_hidden_dims: List[int],
        nonlinearity: str,
        dropout_rate: float,
    ):
        super(VisionGuidedBatchNormMLP, self).__init__()
        self.image_encoder = image_encoder
        self.policy_head = BatchNormMLP(
            input_dim=image_encoder.feature_dim + robot_state_dim,
            hidden_dims=policy_hidden_dims,
            output_dim=action_dim,
            nonlinearity=nonlinearity,
            dropout_rate=dropout_rate,
        )
        for param in list(self.policy_head.parameters())[-2:]:
            param.data = 1e-2 * param.data

    def forward(self, images, point_clouds, robot_states, texts):
        image_emb = self.image_encoder(images)
        emb = torch.cat([image_emb, robot_states], dim=1)
        actions = self.policy_head(emb)
        return actions


class PointCloudGuidedBatchNormMLP(Actor):

    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        robot_state_dim: int,
        action_dim: int,
        policy_hidden_dims: List[int],
        nonlinearity: str,
        dropout_rate: float,
    ):
        super(PointCloudGuidedBatchNormMLP, self).__init__()
        self.point_cloud_encoder = point_cloud_encoder
        self.policy_head = BatchNormMLP(
            input_dim=point_cloud_encoder.feature_dim + robot_state_dim,
            hidden_dims=policy_hidden_dims,
            output_dim=action_dim,
            nonlinearity=nonlinearity,
            dropout_rate=dropout_rate,
        )
        for param in list(self.policy_head.parameters())[-2:]:
            param.data = 1e-2 * param.data

    def forward(self, images, point_clouds, robot_states, texts):
        # * Notice: normalize the input point cloud
        point_clouds = PointCloud.normalize(point_clouds)
        point_cloud_emb = self.point_cloud_encoder(point_clouds)
        emb = torch.cat([point_cloud_emb, robot_states], dim=1)
        actions = self.policy_head(emb)
        return actions
