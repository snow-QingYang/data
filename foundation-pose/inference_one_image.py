# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from PIL import Image
import numpy as np
import trimesh
import imageio
import os
import torch
import nvdiffrast.torch as dr
import math


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--mesh_file",
        type=str,
        default=f"{code_dir}/demo_data/1039382382090.0/1039382382090.0.glb",
    )
    parser.add_argument(
        "--test_image",
        type=str,
        default=f"{code_dir}/demo_data/1039382382090.0/1039382382090.jpg",
    )
    parser.add_argument(
        "--test_depth",
        type=str,
        default=f"{code_dir}/demo_data/1039382382090.0/1039382382090.depth.npy",
    )
    parser.add_argument(
        "--test_mask",
        type=str,
        default=f"{code_dir}/demo_data/1039382382090.0/1039382382090.mask.0.png",
    )
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=3)
    parser.add_argument(
        "--debug_dir", type=str, default=f"{code_dir}/debug/1039382382090.0"
    )
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    if isinstance(mesh, trimesh.Scene):
        for m in mesh.geometry.values():
            mesh = m
            break

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
    h, w, c = color.shape

    f = math.sqrt(w*w + h*h)
    K = np.array(
        [
            [f, 0.0, w / 2],
            [0.0, f, h / 2],
            [0.0, 0.0, 1.0],
        ]
    )

    radius = est.diameter / 2
    vs,us = np.where(mask>0)
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    radius_image = np.sqrt((us - uc) ** 2 + (vs - vc) ** 2).max()

    alpha = f * radius / radius_image / depth[int(vc), int(uc)] 
    depth = depth * alpha


    pose = est.register(
        K=K,
        rgb=color,
        depth=depth,
        ob_mask=mask,
        iteration=args.est_refine_iter,
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

        coord_color = render_scene_coordinate_map(
            mesh=mesh,
            obj_pose=torch.from_numpy(pose).float().cuda(),
            K=K,
            H=h,
            W=w,
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
