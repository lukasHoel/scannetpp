import argparse
import os
import shutil
import json
import sys
from pathlib import Path

import imageio
import numpy as np
from tqdm import tqdm
try:
    import renderpy
except ImportError:
    print("renderpy not installed. Please install renderpy from https://github.com/liu115/renderpy")
    sys.exit(1)

from common.utils.colmap import read_model, write_model, Image
from common.scene_release import ScannetppScene_Release
from common.utils.utils import run_command, load_yaml_munch, load_json, read_txt_list


def sample_cameras_in_mesh(mesh_path):
    import trimesh

    mesh: trimesh.Trimesh = trimesh.load(mesh_path, force='mesh')

    xyz_min, xyz_max = mesh.bounds
    n_voxels = 8**3
    min_camera_distance_to_mesh = 0.1
    n_dir = 4
    voxel_size = np.pow((xyz_max - xyz_min).prod() / n_voxels, 1.0/3.0)
    voxel_reso = ((xyz_max - xyz_min) / voxel_size).astype(np.long)
    core_ratio = np.array([0.6, 0.6, 0.6])
    core_bbox = [xyz_min * core_ratio, xyz_max * core_ratio]
    core_bbox_length = (xyz_max - xyz_min) * core_ratio

    X = np.linspace(0, 1, voxel_reso[0]) * core_bbox_length[0] + core_bbox[0][0]
    Y = np.linspace(0, 1, voxel_reso[1]) * core_bbox_length[1] + core_bbox[0][1]
    Z = np.linspace(0, 1, voxel_reso[2]) * core_bbox_length[2] + core_bbox[0][2]

    grid_x, grid_y, grid_z = np.meshgrid(X, Y, Z, indexing="ij")
    grid_xyz = np.stack([grid_x, grid_y, grid_z], axis=-1)  # get voxelized positions grid_xyz, [N_1, N_2, N_3, 3]
    grid_xyz = np.reshape(grid_xyz, (-1, 3))

    # Filter out inappropriate points
    closest, distance, triangle_id = trimesh.proximity.closest_point(mesh, grid_xyz)
    filtering_mask = distance >= min_camera_distance_to_mesh
    grid_xyz = grid_xyz[filtering_mask]

    # trimesh.PointCloud(vertices=grid_xyz).export("tmp.ply")

    RT_list = []
    for i in tqdm(range(grid_xyz.shape[0])):
        # sample multiple cameras for each point
        camera_pos = grid_xyz[i]

        def pose_spherical(theta, phi, radius):
            trans_t = lambda t: np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, t],
                [0, 0, 0, 1]]).astype(np.float32)

            rot_phi = lambda phi: np.array([
                [1, 0, 0, 0],
                [0, np.cos(phi), -np.sin(phi), 0],
                [0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 1]]).astype(np.float32)

            rot_theta = lambda th: np.array([
                [np.cos(th), 0, -np.sin(th), 0],
                [0, 1, 0, 0],
                [np.sin(th), 0, np.cos(th), 0],
                [0, 0, 0, 1]]).astype(np.float32)

            c2w = trans_t(radius)
            c2w = rot_phi(phi / 180. * np.pi) @ c2w
            c2w = rot_theta(theta / 180. * np.pi) @ c2w
            # c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
            c2w[0:3, 3] = -1 * c2w[0:3, 3]
            return c2w

        theta = np.random.rand(n_dir) * 360
        phi = np.random.rand(n_dir) * 0
        c2w = [pose_spherical(theta[i], phi[i], 1) for i in range(n_dir)]
        c2w = np.stack(c2w, axis=0)  # [n_dir, 4, 4]

        c2w[:, 0:3, 3] = camera_pos
        RT = np.linalg.inv(c2w)
        RT_list.append(RT)

    RT_list = np.concatenate(RT_list, axis=0)

    return RT_list

def sample_noise(n, r_max, t_max):
    nr = np.random.normal(0, scale=r_max/2.0, size=(n,3))
    nr = np.clip(nr, a_min=-r_max, a_max=r_max)

    nt = np.random.normal(0, scale=t_max/2.0, size=(n,3))
    nt = np.clip(nt, a_min=-t_max, a_max=t_max)

    return nr, nt


def interpolate_noise(n, steps):
    last = np.linspace(n[-1], n[-1], num=steps)
    n = [np.linspace(n[i], n[i + 1], num=steps) for i in range(n.shape[0] - 1)]
    n.append(last)
    n = np.concatenate(n, axis=0)
    return n


def to_degrees(x):
    return x * 180.0 / np.pi


def to_radians(x):
    return x * np.pi / 180.0


# Checks if a matrix is a valid rotation matrix.
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-5


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# Calculates Rotation Matrix given euler angles.
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def apply_noise(poses, chunk_size=1, r_max=5.0, t_max=0.05) -> np.array:
    noisy_poses = []

    # create noise vectors
    n = len(poses) // chunk_size + (len(poses) % chunk_size != 0)
    nr, nt = sample_noise(n, r_max, t_max)
    nr = interpolate_noise(nr, chunk_size)
    nt = interpolate_noise(nt, chunk_size)

    for i, p in enumerate(poses):
        pose_numpy = p

        # extract r, t
        r = pose_numpy[:3, :3]
        r = rotationMatrixToEulerAngles(r)
        r = to_degrees(r)
        t = pose_numpy[:3, 3]

        # get noise
        nr_i = nr[i // chunk_size]
        nt_i = nt[i // chunk_size]

        # apply noise
        r += nr_i
        t += nt_i

        # create pose noise
        r = to_radians(r)
        r = eulerAnglesToRotationMatrix(r)
        p_noise = np.eye(4, dtype=np.float32)
        p_noise[:3, :3] = r
        p_noise[:3, 3] = t

        noisy_poses.append(p_noise)

    return noisy_poses


def scale_intrinsics_opencv(fx, fy, cx, cy, h, w, downscale_factor: int):
    fx = fx / downscale_factor
    fy = fy / downscale_factor
    cx = (cx + 0.5) / downscale_factor - 0.5
    cy = (cy + 0.5) / downscale_factor - 0.5
    h = int(h / downscale_factor)
    w = int(w / downscale_factor)
    return fx, fy, cx, cy, h, w


def check_if_scene_complete(scene_id, output_dir, render_devices, save_rendered_rgb, save_rendered_depth, downscale_factor, imu_every_nth_frame, render_novel_views):
    if render_novel_views:
        return False
    
    # check if all is rendered already and could skip
    complete = True
    for device in render_devices:
        if "iphone_imu" in device:
            rgb_dir = Path(output_dir) / scene_id / "iphone" / f"imu_downscaled_{downscale_factor}_render_rgb"
            depth_dir = Path(output_dir) / scene_id / "iphone" / f"imu_downscaled_{downscale_factor}_render_depth"
            imu_file = Path(output_dir) / scene_id / "iphone" / "pose_intrinsic_imu.json"
            imu_file = json.load(open(imu_file))
            src_indices = [i for i in range(len(imu_file.keys())) if i % imu_every_nth_frame == 0]
            num_src_rgb = len(src_indices)
        else:
            image_type = "dslr" if "dslr" in device else "iphone"
            out_dir_prefix = "undistorted_" if "undistorted" in device else ""
            src_folder = "undistorted_images" if "undistorted" in device else "images" if "dslr" in device else "rgb"
            src_rgb_dir = Path(output_dir) / scene_id / image_type / src_folder
            rgb_dir = Path(output_dir) / scene_id / image_type / f"{out_dir_prefix}render_rgb"
            depth_dir = Path(output_dir) / scene_id / image_type / f"{out_dir_prefix}render_depth"
            num_src_rgb = len(os.listdir(str(src_rgb_dir))) if os.path.exists(src_rgb_dir) else 0

        num_rend_rgb = len(os.listdir(str(rgb_dir))) if os.path.exists(rgb_dir) else 0
        num_rend_depth = len(os.listdir(str(depth_dir))) if os.path.exists(depth_dir) else 0
        if (num_src_rgb != num_rend_rgb and save_rendered_rgb) or (num_src_rgb != num_rend_depth and save_rendered_depth):
            complete = False
            break

    return complete

def main(args):
    cfg = load_yaml_munch(args.config_file)

    # get all scenes to process
    if cfg.get("scene_ids"):
        scene_ids = cfg.scene_ids
    elif cfg.get("splits"):
        scene_ids = []
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / "splits" / f"{split}.txt"
            scene_ids += read_txt_list(split_path)

    # setup output dir
    output_dir = cfg.get("output_dir")
    if output_dir is None:
        # default to data folder in data_root
        output_dir = Path(cfg.data_root) / "data"
    output_dir = Path(output_dir)

    # setup render devices
    render_devices = []
    if cfg.get("render_dslr", False):
        render_devices.append("dslr")
    if cfg.get("render_dslr_undistorted", False):
        render_devices.append("dslr_undistorted")
    if cfg.get("render_iphone", False):
        render_devices.append("iphone")
    if cfg.get("render_iphone_undistorted", False):
        render_devices.append("iphone_undistorted")
    if cfg.get("render_iphone_imu", False):
        render_devices.append("iphone_imu")

    # setup render options
    save_rendered_rgb = cfg.get("save_rendered_rgb", True)
    save_rendered_depth = cfg.get("save_rendered_depth", True)
    downscale_factor = cfg.get("downscale_factor", 1)
    imu_every_nth_frame = cfg.get("imu_every_nth_frame", 1)
    render_novel_views = cfg.get("render_novel_views", False)

    # filter remaining scene_ids
    print("Filtering scene ids")
    filtered_scene_ids = [scene_id for scene_id in scene_ids if check_if_scene_complete(scene_id, output_dir, render_devices, save_rendered_rgb, save_rendered_depth, downscale_factor, imu_every_nth_frame, render_novel_views)]
    print(f"{len(scene_ids) - len(filtered_scene_ids)} scenes already rendered, skipping")
    scene_ids = filtered_scene_ids

    # go through each scene
    print("Rendering scenes with offset=", args.offset, "and stride=", args.stride)
    scene_ids = scene_ids[args.offset::args.stride]
    pbar = tqdm(scene_ids, desc="scene")
    executed_iters = 0
    for scene_id in pbar:
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")
        render_engine = renderpy.Render()
        render_engine.setupMesh(str(scene.scan_mesh_path))
        for device in render_devices:
            pbar.set_description_str(f"scene {scene_id}, device {device}")
            if "dslr" in device:
                cameras, images, points3D = read_model(scene.dslr_colmap_dir, ".txt")
            else:
                cameras, images, points3D = read_model(scene.iphone_colmap_dir, ".txt")
                    
            assert len(cameras) == 1, "Multiple cameras not supported"
            camera = next(iter(cameras.values()))

            fx, fy, cx, cy = camera.params[:4]
            params = camera.params[4:]
            camera_model = camera.model
            height = camera.height
            width = camera.width
            orig_height = height
            orig_width = width

            out_dir_prefix = ""
            device_out_dir = device
            if "undistorted" in device:
                dslr_transforms_path = os.path.join(scene.dslr_dir, "nerfstudio", "transforms_undistorted.json")
                iphone_transforms_path = os.path.join(scene.iphone_data_dir, "nerfstudio", "transforms_undistorted.json")
                if "dslr" in device:
                    transforms_path = dslr_transforms_path
                    device_out_dir = "dslr"
                else:
                    transforms_path = iphone_transforms_path
                    device_out_dir = "iphone"
                with open(transforms_path, "r") as f:
                    transforms_undistorted = json.load(f)
                fx = transforms_undistorted["fl_x"]
                fy = transforms_undistorted["fl_y"]
                cx = transforms_undistorted["cx"]
                cy = transforms_undistorted["cy"]
                params[0] = transforms_undistorted["k1"]
                params[1] = transforms_undistorted["k2"]
                params[2] = transforms_undistorted["k3"] if "dslr" in device else transforms_undistorted["p1"]
                params[3] = transforms_undistorted["k4"] if "dslr" in device else transforms_undistorted["p2"]
                height = transforms_undistorted["h"]
                width = transforms_undistorted["w"]
                camera_model = "PINHOLE"
                out_dir_prefix = "undistorted_"
            
            if "imu" in device:
                out_dir_prefix = "imu_"
                device_out_dir = "iphone"
            
            # downscale intrinsics
            if downscale_factor > 1:
                fx, fy, cx, cy, height, width = scale_intrinsics_opencv(
                    fx, fy, cx, cy,
                    height, width,
                    downscale_factor
                )
                out_dir_prefix += f"downscaled_{downscale_factor}_"

            render_engine.setupCamera(
                height, width,
                fx, fy, cx, cy,  # our intrinsics are in opencv convention, so we should shift principal point by +0.5 (because we render with opengl). The render_engine does this internally!
                camera_model,
                params,      # Distortion parameters np.array([k1, k2, k3, k4]) or np.array([k1, k2, p1, p2])
            )

            near = cfg.get("near", 0.05)
            far = cfg.get("far", 20.0)

            if render_novel_views:
                assert "imu" not in device, "Novel view rendering not supported for IMU data"
                # render novel views
                rgb_dir = Path(output_dir) / scene_id / device_out_dir / f"{out_dir_prefix}render_rgb_novel"
                depth_dir = Path(output_dir) / scene_id / device_out_dir / f"{out_dir_prefix}render_depth_novel"
                if os.path.exists(rgb_dir):
                    shutil.rmtree(rgb_dir)
                if os.path.exists(depth_dir):
                    shutil.rmtree(depth_dir)
                rgb_dir.mkdir(parents=True, exist_ok=True)
                depth_dir.mkdir(parents=True, exist_ok=True)


                # RT_list = sample_cameras_in_mesh(str(scene.scan_mesh_path))
                orig_RT = [v.world_to_camera for k,v in images.items()]
                noisy_RT = np.concatenate([
                    apply_noise(orig_RT, r_max=10, t_max=0.05),
                    # apply_noise(orig_RT, r_max=10, t_max=0.05),
                    # apply_noise(orig_RT, r_max=5, t_max=0.15),
                    # apply_noise(orig_RT, r_max=10, t_max=0.15)
                ], axis=0)

                for id, RT in enumerate(tqdm(noisy_RT, f"Rendering {device} images")):
                    world_to_camera = RT
                    name = f"novel_{id:04d}"
                    rgb, depth, vert_indices = render_engine.renderAll(world_to_camera, near, far)
                    rgb = rgb.astype(np.uint8)
                    # Make depth in mm and clip to fit 16-bit image
                    depth = (depth.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)
                    imageio.imwrite(rgb_dir / f"{name}.jpg", rgb)
                    imageio.imwrite(depth_dir / f"{name}.png", depth)

                # save poses in nerfstudio convention
                noisy_c2w = np.linalg.inv(noisy_RT)
                noisy_c2w[:, 0:3, 1:3] *= -1
                noisy_c2w = noisy_c2w[:, np.array([1, 0, 2, 3]), :]
                noisy_c2w[:, 2, :] *= -1
                transforms_noisy = transforms_undistorted
                transforms_noisy.pop("test_frames")
                transforms_noisy["frames"] = [{
                    "file_path": f"novel_{id:04d}.jpg",
                    "transform_matrix": p.tolist()
                } for id, p in enumerate(noisy_c2w)]
                with open(os.path.join(scene.dslr_dir, "nerfstudio", "transforms_noisy.json"), "w") as f:
                    json.dump(transforms_noisy, f, indent=4)

            else:
                # render camera views 
                
                # setup output dirs
                rgb_dir = Path(output_dir) / scene_id / device_out_dir / f"{out_dir_prefix}render_rgb"
                depth_dir = Path(output_dir) / scene_id / device_out_dir / f"{out_dir_prefix}render_depth"
                if save_rendered_rgb:
                    rgb_dir.mkdir(parents=True, exist_ok=True)
                if save_rendered_depth:
                    depth_dir.mkdir(parents=True, exist_ok=True)
                num_rend_rgb = len(os.listdir(str(rgb_dir))) if os.path.exists(rgb_dir) else 0
                num_rend_depth = len(os.listdir(str(depth_dir))) if os.path.exists(depth_dir) else 0
                
                # get frames to render
                image_names = [image.name for image in images.values()]
                image_ext = os.path.splitext(image_names[0])[-1]
                image_names = [os.path.splitext(image_name)[0] for image_name in image_names]
                w2c_list = [image.world_to_camera for image in images.values()]
                if "imu" in device:
                    # load imu poses
                    imu_data = json.load(open(scene.iphone_pose_intrinsic_imu_path))
                    image_names = list(imu_data.keys())
                    image_names = image_names[::imu_every_nth_frame]
                    w2c_list = [np.linalg.inv(imu_data[name]["aligned_pose"]) for name in image_names]

                # check if already rendered
                num_img = len(image_names)
                if (num_img == num_rend_rgb or not save_rendered_rgb) and (num_img == num_rend_depth or not save_rendered_depth):
                    print("already rendered, skip", scene_id, device)
                    continue
                for idx, image_name in enumerate(tqdm(image_names, f"Rendering {device} images")):
                    # setup output file names
                    out_file_name = image_name.split("/")[-1]
                    rgb_out_file_path = rgb_dir / (out_file_name + image_ext)
                    depth_out_file_path = depth_dir / (out_file_name + ".png")

                    # check if already rendered
                    if (rgb_out_file_path.exists() or not save_rendered_rgb) and (depth_out_file_path.exists() or not save_rendered_depth):
                        continue
                    
                    if "imu" in device:
                        # load intrinsics for this frame
                        imu_K = np.array(imu_data[image_name]["intrinsic"])
                        fx = imu_K[0, 0]
                        fy = imu_K[1, 1]
                        cx = imu_K[0, 2]
                        cy = imu_K[1, 2]

                        # our intrinsics are in arkit/opengl convention, but we want opencv convention: subtract 0.5 here.
                        cx = cx - 0.5
                        cy = cy - 0.5

                        # downscale intrinsics
                        if downscale_factor > 1:
                            fx, fy, cx, cy, height, width = scale_intrinsics_opencv(
                                fx, fy, cx, cy,
                                orig_height, orig_width,
                                downscale_factor
                            )
                        
                        # set intrinsics for this frame
                        render_engine.setupCamera(
                            height, width,
                            fx, fy, cx, cy,  # our intrinsics are in opencv convention, so we should shift principal point by +0.5 (because we render with opengl). The render_engine does this internally!
                            "PINHOLE",
                            np.array([0.0, 0.0, 0.0, 0.0]),
                        )

                    # render with camera
                    world_to_camera = w2c_list[idx]
                    rgb, depth, vert_indices = render_engine.renderAll(world_to_camera, near, far)
                    
                    # save rgb
                    if save_rendered_rgb:
                        rgb = rgb.astype(np.uint8)
                        imageio.imwrite(rgb_out_file_path, rgb)

                    # Make depth in mm and clip to fit 16-bit image
                    if save_rendered_depth:
                        depth = (depth.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)
                        imageio.imwrite(depth_out_file_path, depth)

        executed_iters += 1
        if args.max_iter > 0 and executed_iters >= args.max_iter:
            break

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    p.add_argument("--stride", type=int, default=1, help="stride for scene subdivision")
    p.add_argument("--offset", type=int, default=0, help="offset for scene subdivision")
    p.add_argument("--max_iter", type=int, default=-1, help="how many iters to do")
    args = p.parse_args()

    main(args)
