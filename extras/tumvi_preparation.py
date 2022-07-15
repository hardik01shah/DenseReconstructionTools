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
import json

import os
import yaml
import shutil

if __name__=="__main__":
    orig_dir = Path("../../tumvi-dataset/original/")

    sequences = os.listdir(orig_dir)
    sequences.sort()

    name_map = {}
    for i, seq in enumerate(sequences):
        seq_name = seq.replace("dataset-", "")
        seq_name = seq_name.replace("_1024_16", "")

        seq_idx = f"{i:02d}"
        name_map[seq_idx] = seq_name

        os.rename(orig_dir / seq, orig_dir / seq_idx)
        
    with open(Path("../../tumvi-dataset/sequence_names.json"), 'w') as json_file:
            json.dump(name_map, json_file, indent = 4)