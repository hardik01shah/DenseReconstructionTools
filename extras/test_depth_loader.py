from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import cv2

import os
import yaml
import shutil

if __name__=="__main__":
    depth_dir = Path("../../tumvi_data/test/kf_data/keypoints/")
    depth_viz_dir = Path("../../tumvi_data/test/kf_data/keypoints_viz/")
    image_dir = Path("../../tumvi_data/test/mav0/cam0/data/")
    save_dir = Path("../../tumvi_data/test/debug_depth/")

    depth_paths = os.listdir(depth_dir)
    target_image_size = (512, 512)
    depth_map = {}

    times = [tns.replace(".txt", "") for tns in depth_paths]
    
    for tns in times:
        depth_map[tns] = str(depth_viz_dir / f"{tns}.txt")
    times.sort()
    
    for tns in tqdm(times):
        depth = torch.zeros(1, target_image_size[0], target_image_size[1], dtype=torch.float32)
        img = cv2.imread(str(image_dir / f"{tns}.png"))

        with open(depth_map[tns], "r") as f:
            lines = f.readlines()
        
        points_sz = int(lines[0])
        for l in lines[1:]:
            tmp = l.split()
            x, y, i_depth = round(float(tmp[0])), round(float(tmp[1])), float(tmp[2])
            depth[0][y][x] = i_depth
            img = cv2.circle(img, (x, y), 4, (0, 0, 255), 1)
        cv2.imwrite(str(save_dir / f"{tns}.png"), img)
        # break
