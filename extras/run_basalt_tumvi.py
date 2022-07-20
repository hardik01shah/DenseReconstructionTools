from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import pandas as pd
from tqdm import tqdm
import cv2
import json
import argparse
import subprocess
import os
import yaml
import shutil

def execute(cmd):
    proc = subprocess.Popen(cmd)
    proc.wait()

if __name__=="__main__":
    # Initialize parser
    desc = "Python code for running basalt on tumvi sequences"
    parser = argparse.ArgumentParser(description = desc)

    # add arguments
    parser.add_argument("tumvi_path", help="path to tum-vi dataset", type=str)
    parser.add_argument("basalt_path", help="path to basalt directory", type=str)
    
    args = parser.parse_args()

    tumvi_dir = Path(os.path.abspath(args.tumvi_path)) / f"rectified"
    basalt_dir = Path(os.path.abspath(args.basalt_path)) 

    sequences = os.listdir(tumvi_dir)
    sequences.sort()

    basalt_vio_path = basalt_dir / "build/basalt_vio"
    basalt_config_path = basalt_dir / "data/tumvi_512_config.json"
    cmd_list = ["",                     # 0
        "--dataset-path", "",           # 1 2
        "--cam-calib", "",              # 3 4
        "--dataset-type", "euroc",      # 5 6
        "--config-path", "",            # 7 8
        "--show-gui", "0",              # 9 10
        "--keyframe-data", ""]          # 11 12

    for seq in sequences:
        print("=" * 20)
        print(f"Running basalt on sequence {seq}")

        seq_path = tumvi_dir / f"{seq}"
        calib_path = seq_path / f"basalt_calib.json" 
        kf_data_path = seq_path / f"basalt_keyframe_data"   


        bash_cmd = cmd_list.copy()
        bash_cmd[0] = str(basalt_vio_path)
        bash_cmd[2] = str(seq_path) 
        bash_cmd[4] = str(calib_path) 
        bash_cmd[8] = str(basalt_config_path) 
        bash_cmd[12] = str(kf_data_path) 
        print(f"executing {' '.join(bash_cmd)}")
        print()
        execute(bash_cmd)
        print()