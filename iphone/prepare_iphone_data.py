
'''
Download ScanNet++ data

Default: download splits with scene IDs and default files
that can be used for novel view synthesis on DSLR and iPhone images
and semantic tasks on the mesh
'''

import argparse
from pathlib import Path
import yaml
from munch import Munch
from tqdm import tqdm
import json
import sys
import subprocess
import zlib
import numpy as np
import imageio as iio
import lz4.block
import os

from common.scene_release import ScannetppScene_Release
from common.utils.utils import run_command, load_yaml_munch, load_json, read_txt_list


def extract_rgb(scene):
    # rename existing folder
    # if scene.iphone_rgb_dir.exists():
    #     print(f"rename {scene.iphone_rgb_dir} to {scene.iphone_rgb_dir}_old")
    #     os.rename(str(scene.iphone_rgb_dir), str(scene.iphone_rgb_dir) + '_old')

    # delete existing folder
    # if scene.iphone_rgb_dir.exists():
    #     print(f"delete {scene.iphone_rgb_dir}")
    #     import shutil
    #     shutil.rmtree(str(scene.iphone_rgb_dir))

    scene.iphone_rgb_dir.mkdir(parents=True, exist_ok=True)

    # files = [x for x in scene.iphone_rgb_dir.iterdir() if Path(x).is_file()]
    # if len(files) > 0:
    #     print(f"delete {len(files)} existing images in {scene.iphone_rgb_dir}")
    #     for f in files:
    #         os.remove(str(f))

    # cmd = f"ffmpeg -i {scene.iphone_video_path} -r 10 -vf \"scale=iw/2:ih/2\" -start_number 0 -q:v 1 {scene.iphone_rgb_dir}/frame_%06d.jpg"
    cmd = f"ffmpeg -i {scene.iphone_video_path} -start_number 0 -q:v 1 {scene.iphone_rgb_dir}/frame_%06d.jpg"
    run_command(cmd, verbose=True)

    # from torchcodec.decoders import VideoDecoder
    # from PIL import Image
    # device = "cpu"  # or e.g. "cuda" !
    # decoder = VideoDecoder(scene.iphone_video_path, device=device)
    # for idx, img in enumerate(decoder):
    #     img = img.permute(1, 2, 0).cpu().numpy()
    #     Image.fromarray(img).save(f"{scene.iphone_rgb_dir}/frame_{idx:06d}.jpg")
    

def extract_masks(scene):
    scene.iphone_video_mask_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {str(scene.iphone_video_mask_path)} -pix_fmt gray -start_number 0 {scene.iphone_video_mask_dir}/frame_%06d.png"
    run_command(cmd, verbose=True)

def extract_depth(scene):
    # global compression with zlib
    height, width = 192, 256
    sample_rate = 1
    scene.iphone_depth_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(scene.iphone_depth_path, 'rb') as infile:
            data = infile.read()
            data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
            depth = np.frombuffer(data, dtype=np.float32).reshape(-1, height, width)

        for frame_id in tqdm(range(0, depth.shape[0], sample_rate), desc='decode_depth'):
            iio.imwrite(f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png", (depth * 1000).astype(np.uint16))
    # per frame compression with lz4/zlib
    except:
        frame_id = 0
        with open(scene.iphone_depth_path, 'rb') as infile:
            while True:
                size = infile.read(4)   # 32-bit integer
                if len(size) == 0:
                    break
                size = int.from_bytes(size, byteorder='little')
                if frame_id % sample_rate != 0:
                    infile.seek(size, 1)
                    frame_id += 1
                    continue

                # read the whole file
                data = infile.read(size)
                try:
                    # try using lz4
                    data = lz4.block.decompress(data, uncompressed_size=height * width * 2)  # UInt16 = 2bytes
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
                except:
                    # try using zlib
                    data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                    depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
                    depth = (depth * 1000).astype(np.uint16)

                # 6 digit frame id = 277 minute video at 60 fps
                iio.imwrite(f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png", depth)
                frame_id += 1

def main(args):
    cfg = load_yaml_munch(args.config_file)

    # get the scenes to process, specify any one
    if cfg.get('scene_list_file'):
        scene_ids = read_txt_list(cfg.scene_list_file)
    elif cfg.get('scene_ids'):
        scene_ids = cfg.scene_ids
    elif cfg.get('splits'):
        scene_ids = []
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / 'splits' / f'{split}.txt'
            scene_ids += read_txt_list(split_path)

    # get the options to process
    # go through each scene
    scene_ids = scene_ids[args.offset::args.stride]
    pbar = tqdm(scene_ids, desc="scene")
    for scene_id in pbar:
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / 'data')

        if cfg.extract_rgb:
            extract_rgb(scene)

        if cfg.extract_masks:
            extract_masks(scene)

        if cfg.extract_depth:
            extract_depth(scene)

        if cfg.get("only_keep_transforms_frames", False):
            # clean up all frames that do not appear in the transforms file
            transforms = load_json(scene.iphone_nerfstudio_transform_path)
            frames = [f['file_path'] for f in transforms['frames']]
            folders = [scene.iphone_rgb_dir, scene.iphone_depth_dir, scene.iphone_video_mask_dir]
            for folder in folders:
                if not folder.exists():
                    continue
                for f in folder.iterdir():
                    if f.name not in frames:
                        os.remove(str(f))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    p.add_argument("--stride", type=int, default=1, help="stride for scene subdivision")
    p.add_argument("--offset", type=int, default=0, help="offset for scene subdivision")
    args = p.parse_args()

    main(args)
