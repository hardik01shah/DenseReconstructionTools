from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import pandas as pd

import os
import yaml
import shutil

if __name__=="__main__":
    depth_dir = "../../tumvi_data/test/basalt_keyframe_data/keypoints/"
    depth_paths = os.listdir(depth_dir)
    target_image_size = (512, 512)
    depth_map = {}

    times = [tns.replace(".txt", "") for tns in depth_paths]
    
    for tns in times:
        depth_map[tns] = str(Path(depth_dir) / f"{tns}.txt")
    times.sort()
    for tns in times:
        depth = torch.zeros(1, target_image_size[0], target_image_size[1], dtype=torch.float32)
        with open(depth_map[tns], "r") as f:
            lines = f.readlines()
        points_sz = int(lines[0])
        print(tns, points_sz)
        for l in lines[1:]:
            tmp = l.split()
            x, y, i_depth = int(float(tmp[0])), int(float(tmp[1])), float(tmp[2])
            depth[0][y][x] = i_depth
            # np.savetxt('f.txt', depth.numpy().squeeze())
            t_np = depth.numpy().squeeze() #convert to Numpy array
            df = pd.DataFrame(t_np) #convert to a dataframe
            df.to_csv("testfile",index=False) #save to file
        break