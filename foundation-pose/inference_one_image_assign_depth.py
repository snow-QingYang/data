# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
import trimesh
import imageio
import os
import torch
import nvdiffrast.torch as dr
import math
from scipy.optimize import minimize_scalar, minimize, root_scalar
import miniball

from estimater import *
from datareader import *
import argparse
from Utils import transform_pts, to_homo_torch

cvcam_to_glcam = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
).astype(float)


def render_scene_coordinate_map(mesh: trimesh.Trimesh, obj_pose, K, H, W):
    """_summary_

    Args:
        mesh (trimesh.Trimesh): _description_
          @obj_pose: (4,4) torch tensor, openCV camera
    """
    glctx = dr.RasterizeCudaContext()
    mesh_tensors = make_mesh_tensors(mesh)
    pos = mesh_tensors["pos"]
    vnormals = mesh_tensors["vnormals"]
    pos_idx = mesh_tensors["faces"]
    has_tex = "tex" in mesh_tensors

    ob_in_glcams = (
            torch.tensor(cvcam_to_glcam, device="cuda", dtype=torch.float)[None] @ obj_pose
    )
    projection_mat = projection_matrix_from_intrinsics(
        K, height=H, width=W, znear=0.001, zfar=100, window_coords="y_up"
    )
    projection_mat = torch.as_tensor(
        projection_mat.reshape(-1, 4, 4), device="cuda", dtype=torch.float
    )
    mtx = projection_mat @ ob_in_glcams

    pos_homo = to_homo_torch(pos)
    pos_clip = (mtx[:, None] @ pos_homo[None, ..., None])[..., 0]

    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=np.asarray([H, W]))
    mask = rast_out[..., 3] > 0

    pts_cam = transform_pts(pos, ob_in_glcams)
    xyz_map, _ = dr.interpolate(pts_cam, rast_out, pos_idx)
    depth = xyz_map[..., 2]

    color, _ = dr.interpolate(
        (pos - pos.min(dim=0)[0]) / (pos.max(dim=0)[0] - pos.min(dim=0)[0]),
        rast_out,
        pos_idx,
    )
    color = torch.cat([color, mask[..., None].float()], dim=-1)
    return color[0].cpu().numpy()


def remove_outliers_by_distance(points, outlier_fraction=0.2, k=10):
    # 计算每个点到其 k 个最近邻的平均距离
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, _ = nbrs.kneighbors(points)
    avg_distances = distances[:, 1:].mean(axis=1)  # 排除自身距离

    # 根据平均距离排序，保留前 (1 - outlier_fraction) 的点
    threshold = np.percentile(avg_distances, (1 - outlier_fraction) * 100)
    inliers = points[avg_distances <= threshold]
    return inliers


# def find_scaling(P, P_C, R):
#     # P = remove_outliers_by_distance(P, outlier_fraction=0.2)
#     delta = P - P_C  # 计算每个点与中心点的差值
#     A = np.sum(delta[:, :2] ** 2, axis=1)  # (x_i - x_c)^2 + (y_i - y_c)^2

#     s = (P_C[:, 2] - np.sqrt(R**2 - A)) / P[:, 2]
#     s = s[np.logical_not(np.isnan(s))]
#     s = np.sort(s)
#     return np.median(s)
#     # ret = []
#     # for i in range(len(s)):
#     #     distance = (s[i] * P[:, 2] - P_C[:, 2]) ** 2 + A
#     #     if all(distance <= R**2):
#     #         ret.append(s[i])
#     # return ret

# def find_scaling(P, center_depth):
#     P = remove_outliers_by_distance(P, outlier_fraction=0.1)
#     near_depth, far_depth = P[:, 2].min(), P[:, 2].max()
#     median_depth = np.median(P[:, 2])
#     return 1 * center_depth / (0 + median_depth)


# def find_scaling(P, P_C, R):
#     P = remove_outliers_by_distance(P, outlier_fraction=0.1)

#     distances = np.linalg.norm(P - P_C, axis=1)
#     max_dist = np.max(distances)
#     A = np.sum(P[:, 0:1] ** 2 + P[:, 1:2] ** 2 + P[:, 2:3] ** 2, axis=-1)
#     B = np.sum(P[:, 0:1] * P_C[:, 0:1] + P[:, 1:2] * P_C[:, 1:2] + P[:, 2:3] * P_C[:, 2:3], axis=-1)
#     C = np.sum(P_C[:, 0:1] ** 2 + P_C[:, 1:2] ** 2 + P_C[:, 2:3] ** 2, axis=-1)
#     s = np.sort((B - np.sqrt(B**2 - A*(C-R**2))) / A)
#     return s.max()
#     # return np.median(s)


def find_scaling(P, P_C, R):
    P = remove_outliers_by_distance(P, outlier_fraction=0.1)

    size = min(5000, len(P))
    sample_indices = np.random.choice(np.arange(len(P)), size=size, replace=False)
    center, radius_squared = miniball.get_bounding_ball(P[sample_indices])
    scale = R / np.sqrt(radius_squared)
    P2 = P * scale
    A = np.sum(
        (P2[:, 0:1] - P_C[:, 0:1]) ** 2 + (P2[:, 1:2] - P_C[:, 1:2]) ** 2, axis=-1
    )
    T = P_C[:, 2:3] - P2[:, 2:3] - np.sqrt(R ** 2 - A[..., None])

    return scale, T[np.logical_not(np.isnan(T))].max()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--mesh_file",
        type=str,
        # default=f"{code_dir}/demo_data/1005022471625.1/1005022471625.1.glb",
    )
    parser.add_argument(
        "--test_image",
        type=str,
        # default=f"{code_dir}/demo_data/1005022471625.1/1005022471625.jpg",
    )
    parser.add_argument(
        "--test_depth",
        type=str,
        # default=f"{code_dir}/demo_data/1005022471625.1/1005022471625.depth.npy",
    )
    parser.add_argument(
        "--test_mask",
        type=str,
        # default=f"{code_dir}/demo_data/1005022471625.1/1005022471625.mask.1.png",
    )
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=3)
    parser.add_argument(
        "--debug_dir", type=str, 
        # default=f"{code_dir}/debug/1005022471625.1"
    )
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    if isinstance(mesh, trimesh.Scene):
        for m in mesh.geometry.values():
            mesh = m
            break

    mesh_tensors = make_mesh_tensors(mesh)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(
        f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam"
    )

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )
    logging.info("estimator initialization done")

    color = imageio.v2.imread(args.test_image)
    depth = np.load(args.test_depth)
    mask = imageio.v2.imread(args.test_mask) > 255 / 2
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        mask = mask[...,3]
    h, w, c = color.shape
    oh, ow = h, w
    if h * w > 1920 * 1440:
        image_scale = np.sqrt(1920 * 1440 / (h * w))
        color = cv2.resize(color, (int(w * image_scale), int(h * image_scale)))
        depth = cv2.resize(depth, (int(w * image_scale), int(h * image_scale)))
        mask = cv2.resize(
            mask.astype(np.uint8),
            (int(w * image_scale), int(h * image_scale)),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = mask > 0.5
        h, w = color.shape[:2]

    f = math.sqrt(w * w + h * h)
    K = np.array(
        [
            [f, 0.0, w / 2],
            [0.0, f, h / 2],
            [0.0, 0.0, 1.0],
        ]
    )

    radius = est.diameter / 2
    vs, us = np.where(mask > 0)
    uc = (us.min() + us.max()) / 2.0
    vc = (vs.min() + vs.max()) / 2.0
    radius_image = np.sqrt((us - uc) ** 2 + (vs - vc) ** 2).max()

    center_depth = f * radius / radius_image

    # alpha = center_depth / depth[int(vc), int(uc)]
    alpha = center_depth / np.median(depth[mask])
    depth = depth * alpha

    ys_pixel, xs_pixel = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords_pixel = np.stack([xs_pixel, ys_pixel], axis=-1)
    coords_pixel = np.concatenate(
        [coords_pixel, np.ones_like(coords_pixel[..., :1])], axis=-1
    )
    # depth_map_point = (np.linalg.inv(K) @ coords_pixel[mask].T).T * (depth[mask, None] / depth[mask, None].max())
    depth_map_point = (np.linalg.inv(K) @ coords_pixel[mask].T).T * depth[mask, None]
    # depth_map_point = (np.linalg.inv(K) @ coords_pixel[int(vc):int(vc) +1, int(uc):int(uc) + 1, :].reshape(-1, 3).T).T * depth[int(vc):int(vc) +1, int(uc):int(uc) + 1,]

    point_center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * center_depth
    point_center = point_center[:, 0][None]
    alpha, T = find_scaling(depth_map_point, point_center, radius)

    # alpha = center_depth / np.median(depth[mask])
    depth = depth * alpha + T

    pose = est.register(
        K=K,
        rgb=color,
        depth=depth,
        ob_mask=mask,
        iteration=args.est_refine_iter,
        center_depth=center_depth,
    )

    # if debug >= 3:
    #     m = mesh.copy()
    #     m.apply_transform(pose)
    #     m.export(f"{debug_dir}/model_tf.obj")
    #     xyz_map = depth2xyzmap(depth, reader.K)
    #     valid = depth >= 0.001
    #     pcd = toOpen3dCloud(xyz_map[valid], color[valid])
    #     o3d.io.write_point_cloud(f"{debug_dir}/scene_complete.ply", pcd)

    os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
    np.savetxt(f"{debug_dir}/ob_in_cam/cam.txt", pose.reshape(4, 4))

    if debug >= 1:
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(
            color,
            ob_in_cam=center_pose,
            scale=0.1,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        # cv2.imshow('1', vis[...,::-1])
        # cv2.waitKey(1)

    of = math.sqrt(ow * ow + oh * oh)
    oK = np.array(
        [
            [of, 0.0, ow / 2],
            [0.0, of, oh / 2],
            [0.0, 0.0, 1.0],
        ]
    )
    coord_color = render_scene_coordinate_map(
        mesh=mesh,
        obj_pose=torch.from_numpy(pose).float().cuda(),
        K=oK,
        H=oh,
        W=ow,
    )
    imageio.imwrite(
        f"{debug_dir}/scene_coord_map.png", (coord_color * 255).astype(np.uint8)
    )

    image = imageio.v2.imread(args.test_image) / 255.0
    merge_coord_color = (
            image * (1 - coord_color[..., -1:])
            + coord_color[..., :-1] * coord_color[..., -1:]
    )

    imageio.v2.imwrite(
        f"{debug_dir}/merge_scene_coord_map.png",
        (merge_coord_color * 255).astype(np.uint8),
    )

    if debug >= 2:
        os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
        imageio.imwrite(f"{debug_dir}/track_vis/img.png", vis)
