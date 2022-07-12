from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

import os
import yaml
import shutil

if os.path.exists(os.path.abspath(os.path.join(__file__, os.pardir, 'oxford_robotcar'))):
    from data_loader.oxford_robotcar.interpolate_poses import interpolate_poses
from utils import map_fn


class TUMVIDataset(Dataset):

    def __init__(self, dataset_dir, frame_count=2, target_image_size=(256, 512), dilation=1):
        """
        Dataset implementation for TUM VI.
        Images have been rectified.
        """
        self.dataset_dir = Path(dataset_dir)
        self.frame_count = frame_count
        self.dilation = dilation
        self.target_image_size = target_image_size
        
        self.intrinsic_path = self.dataset_dir / "dso/camchain.yaml"
        self.pose_path = self.dataset_dir / "basalt_keyframe_data/poses/keyframeTrajectory_cam.txt"
        self.keypoint_path = self.dataset_dir / "basalt_keyframe_data/keypoints/"
        self.images_path = self.dataset_dir / "mav0/cam0/data/"
        self.debug_path = self.dataset_dir / "monorec_data"

        assert os.path.exists(self.intrinsic_path)
        assert os.path.exists(self.pose_path)
        assert os.path.isdir(self.keypoint_path)
        assert os.path.isdir(self.images_path)

        self._intrinsics, self.orig_image_size = self.load_intrinsics()
        self._pcalib = self.invert_pcalib(np.loadtxt(self.dataset_dir / "dso/cam0/pcalib.txt"))
        (self.pose_times, self._poses, self._rgb_paths) = self.load_data()

        self._offset = (frame_count // 2) * self.dilation
        self._length = len(self.pose_times) - frame_count * dilation
        self._depth = torch.zeros((1, target_image_size[0], target_image_size[1]), dtype=torch.float32)

        self._intrinsics, self.resz_shape, self.crop_box = format_intrinsics(self._intrinsics, self.target_image_size, self.orig_image_size)

    def __getitem__(self, index: int):
        frame_count = self.frame_count
        offset = self._offset

        keyframe_intrinsics = self._intrinsics
        keyframe = self.open_image(index + offset, self.crop_box, index + offset)
        keyframe_pose = self._poses[self.pose_times[index + offset]]
        keyframe_depth = self.open_depth(index + offset)

        frames = [self.open_image(index + i, self.crop_box, index + offset) for i in range(0, (frame_count + 1) * self.dilation, self.dilation) if i != offset]
        intrinsics = [self._intrinsics for _ in range(frame_count)]
        poses = [self._poses[self.pose_times[index + i]] for i in range(0, (frame_count + 1) * self.dilation, self.dilation) if i != offset]

        data = {
            "keyframe": keyframe,
            "keyframe_pose": keyframe_pose,
            "keyframe_intrinsics": keyframe_intrinsics,
            "frames": frames,
            "poses": poses,
            "intrinsics": intrinsics,
            "sequence": torch.tensor([0]),
            "image_id": torch.tensor([index + offset])
        }
        return data, keyframe_depth

    def __len__(self) -> int:
        return self._length

    def load_intrinsics(self):

        stream = open(self.intrinsic_path, "r")
        try:
            cam_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit()
        
        cam0_intrinsics = {
            "fx" : cam_data["cam0"]["intrinsics"][0],
            "fy" : cam_data["cam0"]["intrinsics"][1],
            "cx" : cam_data["cam0"]["intrinsics"][2],
            "cy" : cam_data["cam0"]["intrinsics"][3]
            }

        cam0_intrinsic_mx = np.eye(4)
        cam0_intrinsic_mx[0,0] = cam0_intrinsics["fx"]
        cam0_intrinsic_mx[1,1] = cam0_intrinsics["fy"]
        cam0_intrinsic_mx[0,2] = cam0_intrinsics["cx"]
        cam0_intrinsic_mx[1,2] = cam0_intrinsics["cy"]

        img_size = (cam_data["cam0"]["resolution"][0], cam_data["cam0"]["resolution"][1])
        
        return torch.tensor(cam0_intrinsic_mx, dtype=torch.float32), img_size

    def load_data(self):

        with open(self.pose_path, "r") as f:
            lines = f.readlines()

        data = np.genfromtxt(lines, dtype=np.float64)
        times = [l.split()[0] for l in lines]
        
        assert len(times) == len(data)

        pose_map = {}
        rgb_map = {}
        
        for i in range(len(data)):
            
            ts = torch.tensor(data[i, 1:4])
            qs = torch.tensor(data[i, 4:])
            rs = torch.eye(4)
            rs[:3, :3] = torch.tensor(Rotation.from_quat(qs).as_matrix())
            rs[:3, 3] = ts
            pose = rs.to(torch.float32)

            pose_map[times[i]] = pose
            rgb_map[times[i]] = self.images_path / f"{times[i]}.png"
        
        times.sort()

        print("Generating keyframes...")
        if os.path.isdir(self.debug_path):
            shutil.rmtree(self.debug_path)
        os.makedirs(self.debug_path / "keyframes/all/")
        with open(self.debug_path / "time_index_mapping.txt", 'w') as f: 
            for i, tns in enumerate(times): 
                shutil.copy2(rgb_map[tns], self.debug_path / f"keyframes/all/{i}.png")
                f.write(f'{i}\t{tns}\t{rgb_map[tns]}\t{pose_map[tns]}\n')
        print(f"[+]Generated data at {self.debug_path}")

        return times, pose_map, rgb_map

    # load pcalib
    def invert_pcalib(self, pcalib):
        inv_pcalib = torch.zeros(256, dtype=torch.float32)
        j = 0
        for i in range(256):
            while j < 255 and i + .5 > pcalib[j]:
                j += 1
            inv_pcalib[i] = j
        return inv_pcalib

    def open_image(self, index, crop_box = None, keyframe_index = None):
        
        assert os.path.exists(self._rgb_paths[self.pose_times[index]])

        img = Image.open(self._rgb_paths[self.pose_times[index]])
        img = img.convert('RGB')

        if crop_box:
            img = img.resize((self.resz_shape[1], self.resz_shape[0]), resample=Image.BILINEAR)
            img = img.crop(crop_box)

        if keyframe_index:
            kf_dir = self.debug_path / f"keyframes/{keyframe_index}"
            if(not os.path.isdir(kf_dir)):
                os.mkdir(kf_dir)
            img.save(f'{kf_dir}/{index}.png')

        image_tensor = torch.tensor(np.array(img)).to(dtype=torch.float32)
        image_tensor = self._pcalib[image_tensor.to(dtype=torch.long)]
        image_tensor = image_tensor / 255 - .5
        if len(image_tensor.shape) == 2:
            image_tensor = torch.stack((image_tensor, image_tensor, image_tensor))
        else:
            image_tensor = image_tensor.permute(2, 0, 1)

        return image_tensor

        # for rectification check: 
        # https://github.com/tum-vision/mono_dataset_code/blob/master/src/BenchmarkDatasetReader.h
    
    def open_depth(self, index):
        return self._depth

def format_intrinsics(intrinsics, target_image_size, orig_image_size):
    fx = intrinsics[0,0].item()
    fy = intrinsics[1,1].item()
    cx = intrinsics[0,2].item()
    cy = intrinsics[1,2].item()

    (kitti_h, kitti_w) = (256, 512)
    (h, w) = target_image_size
    (orig_h, orig_w) = orig_image_size

    fx_kitti = 489.2307
    fy_kitti = 489.2307   

    """
    w_crop = (fx * kitti_w)//fx_kitti
    h_crop = (fy * kitti_h)//fy_kitti

    scale_w = w / w_crop
    fx_new = fx * scale_w

    scale_h = h / h_crop
    fy_new = fy * scale_h

    cx_new = (cx - (orig_w - w_crop)/2) * scale_w
    cy_new = (cy - (orig_h - h_crop)/2) * scale_h

    """

    scale_w = fx_kitti / fx
    scale_h = fy_kitti / fy
    
    w_new = int(scale_w * orig_w)
    h_new = int(scale_h * orig_h)

    scale_w = w_new / orig_w
    scale_h = h_new / orig_h

    fx_new = fx * scale_w
    fy_new = fy * scale_h
    cx_new = (cx + 0.5)*scale_w - 0.5 - (w_new - w)//2
    cy_new = (cy + 0.5)*scale_h - 0.5 - (h_new - h)//2

    intrinsics_new = intrinsics.clone()
    intrinsics_new[0,0] = fx_new
    intrinsics_new[1,1] = fy_new
    intrinsics_new[0,2] = cx_new
    intrinsics_new[1,2] = cy_new

    box = ((w_new - w)//2, (h_new - h)//2, (w_new - w)//2 + w, (h_new - h)//2 + h)

    print(f"Orig Intrinsics: {intrinsics}")
    print(f"New Intrinsics: {intrinsics_new}")
    print(f"Scale_w: {scale_w}")
    print(f"Scale_h: {scale_h}")
    # print(f"w_crop: {w_crop}")
    # print(f"h_crop: {h_crop}")
    print(f"w_new: {w_new}")
    print(f"h_new: {h_new}")
    print(box)

    return intrinsics_new, (w_new, h_new), box