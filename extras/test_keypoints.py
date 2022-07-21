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
    tumvi_dir = Path("../../tumvi-dataset/rectified/")
    sequences = os.listdir(tumvi_dir)
    sequences.sort()
        
    for seq in sequences:
        depth_dir = tumvi_dir / f"{seq}/basalt_keyframe_data/keypoints/"
        depth_viz_dir = tumvi_dir / f"{seq}/basalt_keyframe_data/keypoints_viz/"
        image_dir = tumvi_dir / f"{seq}/mav0/cam0/data/"
        save_dir = tumvi_dir / f"{seq}/debug_depth/"
        
        if(os.path.isdir(save_dir)):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)

        depth_paths = os.listdir(depth_dir)
        target_image_size = (512, 512)
        depth_map = {}

        times = [tns.replace(".txt", "") for tns in depth_paths]
        
        for tns in times:
            depth_map[tns] = str(depth_viz_dir / f"{tns}.txt")
            assert os.path.exists(depth_map[tns])
        times.sort()
        
        for tns in tqdm(times):
            depth = torch.zeros(1, target_image_size[0], target_image_size[1], dtype=torch.float32)
            img = cv2.imread(str(image_dir / f"{tns}.png"))

            with open(depth_map[tns], "r") as f:
                lines = f.readlines()
            
            points_sz = int(lines[0])
            for l in lines[1:]:
                tmp = l.split()
                try:
                    x, y, i_depth = round(float(tmp[0])), round(float(tmp[1])), float(tmp[2])
                except:
                    print(f"Error in reading text file: {seq} {tns} {x} {y}\n{tmp[0]} {tmp[1]} {tmp[2]}")
                    continue   

                if not (y>=0 and y<=511 and x>=0 and x<=511 and i_depth>=0):
                    print(f"Error in dimensions:\n{seq} {tns} {x} {y} {i_depth}")
                else:
                    depth[0][y][x] = i_depth
                    img = cv2.circle(img, (x, y), 4, (0, 0, 255), 1)
    
            cv2.imwrite(str(save_dir / f"{tns}.png"), img)
            # break
