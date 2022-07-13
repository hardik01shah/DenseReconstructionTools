import time
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import torch

import sys
sys.path.append("..")

from data_loader.tum_vi_dataset import TUMVIDataset
from model.monorec.monorec_model import MonoRecModel
from utils import unsqueezer, map_fn, to

 # Initialize parser
desc = "Test forward pass for Monorec on TUM-VI dataset."
parser = argparse.ArgumentParser(description = desc)

# add arguments
parser.add_argument("dataset_path", help="path to tum-vi sequence", type=str)
args = parser.parse_args()

target_image_size = (256, 512)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
dataset = TUMVIDataset(dataset_dir = args.dataset_path, target_image_size = target_image_size)

checkpoint_location = Path("../saved/checkpoints/monorec_depth_ref.pth")

inv_depth_min_max = [0.33, 0.0025]

print("Initializing model...")
monorec_model = MonoRecModel(checkpoint_location=checkpoint_location, inv_depth_min_max=inv_depth_min_max)

monorec_model.to(device)
monorec_model.eval()

print("Fetching data...")
index = 464

batch, depth = dataset.__getitem__(index)
batch = map_fn(batch, unsqueezer)
# depth = map_fn(depth, unsqueezer)

batch = to(batch, device)

print("Starting inference...")
s = time.time()
with torch.no_grad():
    data = monorec_model(batch)

prediction = data["result"][0, 0].cpu()
mask = data["cv_mask"][0, 0].cpu()
# depth = depth[0, 0].cpu()

e = time.time()
print(f"Inference took {e - s}s")

plt.imsave("depth.png", prediction.detach().squeeze())
plt.imsave("mask.png", mask.detach().squeeze())
plt.imsave("kf.png", batch["keyframe"][0].permute(1, 2, 0).cpu().numpy() + 0.5)

plt.title(f"MonoRec (took {e - s}s)")
plt.imshow(prediction.detach().squeeze(), vmin=1 / 80, vmax=1 / 5)
plt.show()
plt.imshow(mask.detach().squeeze())
plt.show()
plt.imshow(batch["keyframe"][0].permute(1, 2, 0).cpu() + .5)
plt.show()
