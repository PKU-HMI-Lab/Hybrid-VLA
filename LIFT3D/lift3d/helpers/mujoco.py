import mujoco
import numpy as np
import open3d as o3d
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from scipy.spatial.transform import Rotation

from lift3d.helpers.graphics import HomogeneousCoordinates


def camera_name_to_id(mujoco_model, camera_name):
    camera_id = mujoco.mj_name2id(
        mujoco_model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    )
    return camera_id


def depth_to_meters(depth, mujoco_model):
    extent = mujoco_model.stat.extent
    near = mujoco_model.vis.map.znear * extent
    far = mujoco_model.vis.map.zfar * extent
    image = near / (1 - depth * (1 - near / far))
    return image


def generate_point_cloud(mujoco_renderer: MujocoRenderer, camera_names):
    o3d_point_clouds, depths = [], []
    mujoco_model = mujoco_renderer.model
    width, height = mujoco_renderer.width, mujoco_renderer.height
    for camera_name in camera_names:
        camera_id = camera_name_to_id(mujoco_model, camera_name)
        viewer = mujoco_renderer._get_viewer(render_mode="rgb_array")
        aspect_ratio = width / height
        fovy = np.radians(mujoco_model.cam_fovy[camera_id])
        fovx = 2 * np.arctan(np.tan(fovy / 2) * aspect_ratio)
        fx, fy = width / (2 * np.tan(fovx / 2)), height / (2 * np.tan(fovy / 2))
        cx, cy = width / 2, height / 2
        o3d_camera_matrix = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )

        image = viewer.render(render_mode="rgb_array", camera_id=camera_id)
        depth = viewer.render(render_mode="depth_array", camera_id=camera_id)
        # image = np.flip(image, axis=0)
        # depth = np.flip(depth, axis=0)
        depth = depth_to_meters(depth, mujoco_model)
        depths.append(depth)

        o3d_depth = o3d.geometry.Image(depth)
        o3d_point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth, o3d_camera_matrix
        )
        o3d_point_cloud.colors = o3d.utility.Vector3dVector(image.reshape(-1, 3) / 255)

        cam_body_id = mujoco_model.cam_bodyid[camera_id]
        cam_pos = mujoco_model.body_pos[cam_body_id]
        # cam_pos = mujoco_model.cam_pos[camera_id]

        c2b_r = np.array(mujoco_model.cam_mat0[camera_id]).reshape((3, 3))
        b2w_r = Rotation.from_quat([0, 1, 0, 0], scalar_first=True).as_matrix()
        c2w_r = np.matmul(c2b_r, b2w_r)
        c2w = HomogeneousCoordinates.pos_rot_to_matrix(cam_pos, c2w_r)
        transformed_cloud = o3d_point_cloud.transform(np.linalg.inv(c2w))
        o3d_point_clouds.append(transformed_cloud)

    combined_cloud = o3d.geometry.PointCloud()
    for point_cloud in o3d_point_clouds:
        combined_cloud += point_cloud
    depths = np.array(depths).squeeze()

    points_array = np.asarray(combined_cloud.points)
    colors_array = (np.asarray(combined_cloud.colors) * 255).astype(np.uint8)
    point_cloud_array = np.hstack((points_array, colors_array))
    return point_cloud_array, depths
