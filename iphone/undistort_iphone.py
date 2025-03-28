import argparse
import os
import tempfile
from pathlib import Path
import json
from copy import deepcopy

import numpy as np
import cv2
from tqdm import tqdm

from common.scene_release import ScannetppScene_Release
from common.utils.utils import load_yaml_munch, load_json, read_txt_list


def compute_undistort_intrinsic(K, height, width, distortion_params):
    assert len(distortion_params.shape) == 1
    assert distortion_params.shape[0] == 4  # iphone transforms file has k1, k2, p1, p2

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K,
        distortion_params,
        (width, height),
        0,
        (width, height),
        centerPrincipalPoint=True
    )
    x, y, w, h = roi
    if x > 1 or y > 1:
        raise ValueError("The undistorted image will be cropped. This is not expected.", roi)
    if w < (width - 1) or h < (height - 1):
        raise ValueError("The undistorted image will be cropped. This is not expected.", roi)
    # Make the cx and cy to be the center of the image. 
    # when using centerPrincipalPoint=True, the cx and cy are already at the center, but the convention is that (0, 0) is the center of the top-left pixel.
    # instead, we want the convention that (0, 0) is the top-left corner of the image-plane. e.g. change 255.5 to 256.0
    new_K[0, 2] = width / 2.0
    new_K[1, 2] = height / 2.0
    return new_K, roi


def undistort_frames(
    frames,
    K,
    height,
    width,
    distortion_params,
    input_image_dir,
    out_image_dir,
):
    new_K, roi = compute_undistort_intrinsic(K, height, width, distortion_params)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, distortion_params, np.eye(3), new_K, (width, height), cv2.CV_32FC1
    )

    if input_image_dir.exists():
        for frame in tqdm(frames, desc="frame"):
            image_path = Path(input_image_dir) / frame["file_path"]
            if not image_path.exists():
                continue
            image = cv2.imread(str(image_path))
            undistorted_image = cv2.remap(
                image,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                # borderMode=cv2.BORDER_REFLECT_101,
            )

            # crop the image
            # x, y, w, h = roi
            # undistorted_image = undistorted_image[y:y+h, x:x+w]

            out_image_path = Path(out_image_dir) / frame["file_path"]
            out_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_image_path), undistorted_image)
    return new_K


def update_transforms_json(transforms, new_K, new_height, new_width):
    new_transforms = deepcopy(transforms)
    new_transforms["h"] = new_height
    new_transforms["w"] = new_width
    new_transforms["fl_x"] = new_K[0, 0]
    new_transforms["fl_y"] = new_K[1, 1]
    new_transforms["cx"] = new_K[0, 2]
    new_transforms["cy"] = new_K[1, 2]
    # The undistortion will be PINHOLE and have no distortion paramaters
    new_transforms["camera_model"] = "PINHOLE"
    for key in ("k1", "k2", "p1", "p2"):
        if key in new_transforms:
            new_transforms[key] = 0.0
    for f in new_transforms["frames"]:
        f.pop("mask_path", None)
    if "test_frames" in new_transforms:
        for f in new_transforms["test_frames"]:
            f.pop("mask_path", None)
    return new_transforms


def main(args):
    cfg = load_yaml_munch(args.config_file)

    # get the scenes to process
    if cfg.get("scene_ids"):
        scene_ids = cfg.scene_ids
    elif cfg.get("splits"):
        scene_ids = []
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / "splits" / f"{split}.txt"
            scene_ids += read_txt_list(split_path)

    # get the options to process
    # go through each scene
    scene_ids = scene_ids[args.offset::args.stride]
    pbar = tqdm(scene_ids, desc="scene")
    for scene_id in pbar:
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")
        input_image_dir = cfg.get("input_image_dir", None)
        if input_image_dir is None:
            input_image_dir = scene.iphone_rgb_dir
        else:
            input_image_dir = scene.iphone_data_dir / input_image_dir

        input_transforms_path = cfg.get("input_transforms_path", None)
        if input_transforms_path is None:
            input_transforms_path = scene.iphone_nerfstudio_transform_path
        else:
            input_transforms_path = scene.iphone_data_dir / input_transforms_path

        out_image_dir = scene.iphone_data_dir / cfg.out_image_dir
        out_transforms_path = scene.iphone_data_dir / cfg.out_transforms_path

        transforms = load_json(input_transforms_path)

        if os.path.exists(str(out_transforms_path)):
            try:
                x = load_json(out_transforms_path)
                if "frames" in x and len(x["frames"]) == len(transforms["frames"]) and out_image_dir.exists():
                    print(f"skip {scene_id}, already undistorted")
                    continue
                else:
                    print(f"delete {str(out_transforms_path)}, incomplete file")
                    os.remove(str(out_transforms_path))
            except:
                print(f"delete {str(out_transforms_path)}, invalid file")
                os.remove(str(out_transforms_path))

        assert len(transforms["frames"]) > 0
        frames = deepcopy(transforms["frames"])
        if "test_frames" not in transforms:
            print(f"{scene_id} has no test split")
        elif not (input_image_dir / transforms["test_frames"][0]["file_path"]).exists():
            print(
                f"{scene_id} test image not found. Might due to the scene belonging to testing scenes. "
                "The resizing will skip those images."
            )
        else:
            assert len(transforms["test_frames"]) > 0
            frames += transforms["test_frames"]

        height = int(transforms["h"])
        width = int(transforms["w"])
        distortion_params = np.array(
            [
                float(transforms["k1"]),
                float(transforms["k2"]),
                float(transforms["p1"]),
                float(transforms["p2"]),
            ]
        )
        fx = float(transforms["fl_x"])
        fy = float(transforms["fl_y"])
        cx = float(transforms["cx"])
        cy = float(transforms["cy"])
        K = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )

        new_K = undistort_frames(
            frames,
            K,
            height,
            width,
            distortion_params,
            input_image_dir,
            out_image_dir,
        )
        new_trasforms = update_transforms_json(transforms, new_K, height, width)
        out_transforms_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_transforms_path, "w") as f:
            json.dump(new_trasforms, f, indent=4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    p.add_argument("--stride", type=int, default=1, help="stride for scene subdivision")
    p.add_argument("--offset", type=int, default=0, help="offset for scene subdivision")
    args = p.parse_args()

    main(args)
