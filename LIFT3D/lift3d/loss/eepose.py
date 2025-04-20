import torch
import torch.nn.functional as F


def check_shapes(preds: torch.Tensor, actions: torch.Tensor, rotation_type: str):
    """Check if the shapes of the predictions and actions match and if the last dimension of the predictions is correct.

    Args:
        preds (torch.Tensor): Predictions tensor.
        actions (torch.Tensor): Actions tensor.
        rotation_type (str): Type of rotation representation. Either 'quaternion' or 'euler'.
    """
    if rotation_type not in ["quaternion", "euler"]:
        raise ValueError(
            f'Expected rotation_type to be either "quaternion" or "euler". Got {rotation_type}'
        )

    if preds.shape != actions.shape:
        raise ValueError(
            f"Shapes of preds and actions must match. Got {preds.shape} and {actions.shape}"
        )

    if preds.ndim != 2:
        raise ValueError(f"Expected preds to be 2D. Got {preds.ndim}D")

    # quaternion: x, y, z, qx, qy, qz, qw, gripper
    # euler angles: x, y, z, roll, pitch, yaw, gripper
    expected_dim = 8 if rotation_type == "quaternion" else 7

    if preds.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected last dimension of preds to be {expected_dim}. Got {preds.shape[-1]}"
        )


def compute_loss_mse(
    preds: torch.Tensor,
    actions: torch.Tensor,
    weight_arm: float = 10.0,
    weight_gripper: float = 0.01,
):
    """Compute the loss between the predicted and actual actions using the MSE loss."""
    check_shapes(preds, actions, "quaternion")
    loss_arm_eepose = weight_arm * F.mse_loss(preds[:, :-1], actions[:, :-1])
    loss_gripper = weight_gripper * F.mse_loss(preds[:, -1:], actions[:, -1:])
    loss = loss_arm_eepose + loss_gripper
    loss_dict = {
        "loss_arm_eepose": loss_arm_eepose,
        "loss_gripper": loss_gripper,
    }
    return loss, loss_dict


def compute_loss_cosine_similarity_quaternion(
    preds: torch.Tensor,
    actions: torch.Tensor,
    weight_arm_position: float = 10.0,
    weight_arm_rotation: float = 0.1,
    weight_gripper: float = 0.01,
):
    """Compute the loss between the predicted and actual actions using cosine similarity for the quaternion representation."""
    check_shapes(preds, actions, "quaternion")
    loss_arm_position = weight_arm_position * F.mse_loss(preds[:, :3], actions[:, :3])
    loss_arm_rotation = weight_arm_rotation * (
        1 - F.cosine_similarity(preds[:, 3:7], actions[:, 3:7]).mean()
    )
    loss_gripper = weight_gripper * F.mse_loss(preds[:, -1:], actions[:, -1:])
    loss = loss_arm_position + loss_arm_rotation + loss_gripper
    loss_dict = {
        "loss_arm_position": loss_arm_position,
        "loss_arm_rotation": loss_arm_rotation,
        "loss_gripper": loss_gripper,
    }
    return loss, loss_dict


def compute_loss_cosine_similarity_euler(
    preds: torch.Tensor,
    actions: torch.Tensor,
    weight_arm_position: float = 10.0,
    weight_arm_rotation: float = 1.0,
    weight_gripper: float = 0.01,
):
    """Compute the loss between the predicted and actual actions using cosine similarity for the euler angles representation."""
    check_shapes(preds, actions, "euler")
    loss_arm_position = weight_arm_position * F.mse_loss(preds[:, :3], actions[:, :3])
    loss_arm_rotation = weight_arm_rotation * (
        1 - F.cosine_similarity(preds[:, 3:6], actions[:, 3:6]).mean()
    )
    loss_gripper = weight_gripper * F.mse_loss(preds[:, -1:], actions[:, -1:])
    loss = loss_arm_position + loss_arm_rotation + loss_gripper
    loss_dict = {
        "loss_arm_position": loss_arm_position,
        "loss_arm_rotation": loss_arm_rotation,
        "loss_gripper": loss_gripper,
    }
    return loss, loss_dict


def compute_loss_mse_rotation_maxtrix(
    preds: torch.Tensor,
    actions: torch.Tensor,
    weight_arm_position: float = 10.0,
    weight_arm_rotation: float = 1.0,
    weight_gripper: float = 0.01,
):
    """Compute the loss between the predicted and actual actions using the MSE loss for the rotation matrix representation.

    Notice: the rotation representation should be quaternion.
    """
    check_shapes(preds, actions, "quaternion")
    quat_preds = preds[:, 3:7]
    quat_actions = actions[:, 3:7]

    def batch_quat_to_rot_torch(batch_quat_tensor: torch.Tensor):
        qx, qy, qz, qw = (
            batch_quat_tensor[:, 0],
            batch_quat_tensor[:, 1],
            batch_quat_tensor[:, 2],
            batch_quat_tensor[:, 3],
        )
        batch_rot_tensor = torch.zeros(batch_quat_tensor.shape[0], 9)
        batch_rot_tensor[:, 0] = 1 - 2 * (qy**2 + qz**2)
        batch_rot_tensor[:, 1] = 2 * (qx * qy - qz * qw)
        batch_rot_tensor[:, 2] = 2 * (qx * qz + qy * qw)
        batch_rot_tensor[:, 3] = 2 * (qx * qy + qz * qw)
        batch_rot_tensor[:, 4] = 1 - 2 * (qx**2 + qz**2)
        batch_rot_tensor[:, 5] = 2 * (qy * qz - qx * qw)
        batch_rot_tensor[:, 6] = 2 * (qx * qz - qy * qw)
        batch_rot_tensor[:, 7] = 2 * (qy * qz + qx * qw)
        batch_rot_tensor[:, 8] = 1 - 2 * (qx**2 + qy**2)
        return batch_rot_tensor

    rot_preds = batch_quat_to_rot_torch(quat_preds)
    rot_actions = batch_quat_to_rot_torch(quat_actions)

    loss_arm_position = weight_arm_position * F.mse_loss(preds[:, :3], actions[:, :3])
    loss_arm_rotation = weight_arm_rotation * F.mse_loss(rot_preds, rot_actions)
    loss_gripper = weight_gripper * F.mse_loss(preds[:, -1:], actions[:, -1:])
    loss = loss_arm_position + loss_arm_rotation + loss_gripper
    loss_dict = {
        "loss_arm_position": loss_arm_position,
        "loss_arm_rotation": loss_arm_rotation,
        "loss_gripper": loss_gripper,
    }
    return loss, loss_dict
