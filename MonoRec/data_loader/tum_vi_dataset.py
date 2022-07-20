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

    def __init__(self, dataset_dir, frame_count=2, sequences=None, depth_folder=None,
                 target_image_size=(256, 512), max_length=None, dilation=1, offset_d=0, use_color=False, 
                 basalt_depth=True, return_stereo=False, return_mvobj_mask=False):
        """
        Dataset implementation for TUM VI.
        Images have been rectified.
        :param dataset_dir: Top level folder for KITTI Odometry (should contain folders sequences, poses, poses_dvso (if available)
        :param frame_count: Number of frames used per sample (excluding the keyframe). By default, the keyframe is in the middle of those frames. (Default=2)
        :param sequences: Which sequences to use. Should be tuple of strings, e.g. ("00", "01", ...)
        :param depth_folder: The folder within the sequence folder that contains the depth information (e.g. sequences/00/{depth_folder})
        :param target_image_size: Desired image size (correct processing of depths is only guaranteed for default value). (Default=(256, 512))
        :param max_length: Maximum length per sequence. Useful for splitting up sequences and testing. (Default=None)
        :param dilation: Spacing between the frames (Default 1)
        :param offset_d: Index offset for frames (offset_d=0 means keyframe is centered). (Default=0)
        :param use_color: Use color (camera 2) or greyscale (camera 0) images (default=True)
        :param basalt_depth: Use depth information from basalt. (Default=False)
        :param return_stereo: Return additional stereo frame. Only used during training. (Default=False)
        :param return_mvobj_mask: Return additional moving object mask. Only used during training. If return_mvobj_mask=2, then the mask is returned as target instead of the depthmap. (Default=False)
        """
        self.dataset_dir = Path(dataset_dir)
        self.frame_count = frame_count
        self.sequences = sequences
        self.dilation = dilation
        self.target_image_size = target_image_size
        self.basalt_depth = basalt_depth
        self.return_stereo = return_stereo

        self.debug = False

        if self.sequences is None:
            self.sequences = [f"{i:02d}" for i in range(28)]

        self._offset = (frame_count // 2) * self.dilation
        extra_frames = frame_count * dilation

        self.cam0_intrinsics = {}
        self.cam1_intrinsics = {}
        self.T_cam1_cam0 = {}
        self.orig_image_size = {}
        self.resz_shape = {}
        self.crop_box = {}

        self.cam0_pcalib = {}
        self.cam1_pcalib = {}
        
        self.pose_times = {}
        self._poses = {}
        self.cam0_paths = {}
        self.cam1_paths = {}
        self.depth_paths = {}
        self._dataset_sizes = {}

        dataset_len = 0
        for seq in self.sequences:

            cam0_i, cam1_i, img_size = self.load_intrinsics(seq)
            self.orig_image_size[seq] = img_size

            cam0_i, cam1_i, _resz_shape, _box = format_intrinsics(cam0_i, cam1_i, self.target_image_size, img_size)
            self.cam0_intrinsics[seq] = cam0_i
            self.cam1_intrinsics[seq] = cam1_i
            self.resz_shape[seq] = _resz_shape
            self.crop_box[seq] = _box

            self.cam0_pcalib[seq] = self.invert_pcalib(self.dataset_dir / f"{seq}/dso/cam0/pcalib.txt")
            self.cam1_pcalib[seq] = self.invert_pcalib(self.dataset_dir / f"{seq}/dso/cam1/pcalib.txt")

            times, poses, cam0_pths, cam1_pths, depth_pths = self.load_data(seq)
            self.pose_times[seq] = times
            self._poses[seq] = poses
            self.cam0_paths[seq] = cam0_pths
            self.cam1_paths[seq] = cam1_pths
            self.depth_paths[seq] = depth_pths
            
            self._dataset_sizes[seq] = len(times) - extra_frames
            dataset_len += len(times) - extra_frames

        
        self._length = dataset_len
        self._depth = torch.zeros((1, target_image_size[0], target_image_size[1]), dtype=torch.float32)

    def __getitem__(self, index: int):
        seq, index = self.get_dataset_index(index)
        frame_count = self.frame_count
        offset = self._offset

        keyframe_intrinsics = self.cam0_intrinsics[seq]
        keyframe = self.open_image(index + offset, seq, self.crop_box[seq], index + offset)
        keyframe_pose = self._poses[seq][self.pose_times[seq][index + offset]]

        if self.basalt_depth:
            keyframe_depth = self.open_depth(index + offset, seq, self.crop_box[seq])
        else:
            keyframe_depth = self._depth

        frames = [self.open_image(index + i, seq, self.crop_box[seq], index + offset) for i in range(0, (frame_count + 1) * self.dilation, self.dilation) if i != offset]
        intrinsics = [self.cam0_intrinsics[seq] for _ in range(frame_count)]
        poses = [self._poses[seq][self.pose_times[seq][index + i]] for i in range(0, (frame_count + 1) * self.dilation, self.dilation) if i != offset]

        data = {
            "keyframe": keyframe,
            "keyframe_pose": keyframe_pose,
            "keyframe_intrinsics": keyframe_intrinsics,
            "frames": frames,
            "poses": poses,
            "intrinsics": intrinsics,
            "sequence": torch.tensor([int(seq)], dtype=torch.int32),
            "image_id": torch.tensor([index + offset], dtype=torch.int32)
        }

        if self.return_stereo:
            stereoframe = self.open_image(index + offset, seq, self.crop_box[seq], keyframe_index=None, stereo=True)

            # keyframe_pose = T_world_cam0
            # stereoframe_pose = T_world_cam1 = T_world_cam0 @ inv(T_cam0_cam1)
            stereoframe_pose = keyframe_pose @ torch.inverse(self.T_cam1_cam0[seq])
            
            data["stereoframe"] = stereoframe
            data["stereoframe_pose"] = stereoframe_pose
            data["stereoframe_intrinsics"] = self.cam1_intrinsics[seq]

        return data, keyframe_depth

    def __len__(self) -> int:
        return self._length
    
    def get_dataset_index(self, index: int):
        for seq, seq_len in self._dataset_sizes.items():
            if index >= seq_len:
                index = index - seq_len
            else:
                return seq, index
        return None, None

    def load_intrinsics(self, seq):

        self.intrinsic_path = self.dataset_dir / f"{seq}/dso/camchain.yaml"
        assert os.path.exists(self.intrinsic_path)

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
        cam1_intrinsics = {
            "fx" : cam_data["cam1"]["intrinsics"][0],
            "fy" : cam_data["cam1"]["intrinsics"][1],
            "cx" : cam_data["cam1"]["intrinsics"][2],
            "cy" : cam_data["cam1"]["intrinsics"][3]
            }

        cam0_intrinsic_mx = np.eye(4)
        cam0_intrinsic_mx[0,0] = cam0_intrinsics["fx"]
        cam0_intrinsic_mx[1,1] = cam0_intrinsics["fy"]
        cam0_intrinsic_mx[0,2] = cam0_intrinsics["cx"]
        cam0_intrinsic_mx[1,2] = cam0_intrinsics["cy"]

        cam1_intrinsic_mx = np.eye(4)
        cam1_intrinsic_mx[0,0] = cam1_intrinsics["fx"]
        cam1_intrinsic_mx[1,1] = cam1_intrinsics["fy"]
        cam1_intrinsic_mx[0,2] = cam1_intrinsics["cx"]
        cam1_intrinsic_mx[1,2] = cam1_intrinsics["cy"]

        assert cam_data["cam0"]["resolution"] == cam_data["cam1"]["resolution"]

        img_size = (cam_data["cam0"]["resolution"][0], cam_data["cam0"]["resolution"][1])
        
        self.T_cam1_cam0[seq] = torch.tensor(np.array(cam_data["cam1"]["T_cn_cnm1"]), dtype=torch.float32)

        return torch.tensor(cam0_intrinsic_mx, dtype=torch.float32), torch.tensor(cam1_intrinsic_mx, dtype=torch.float32), img_size

    def load_data(self, seq):

        pose_path = self.dataset_dir / f"{seq}/basalt_keyframe_data/poses/keyframeTrajectory_cam.txt"
        keypoint_path = self.dataset_dir / f"{seq}/basalt_keyframe_data/keypoints_viz/"
        cam0_images_path = self.dataset_dir / f"{seq}/mav0/cam0/data/"
        cam1_images_path = self.dataset_dir / f"{seq}/mav0/cam1/data/"

        assert os.path.exists(pose_path)
        assert os.path.isdir(keypoint_path)
        assert os.path.isdir(cam0_images_path)
        assert os.path.isdir(cam1_images_path)

        with open(pose_path, "r") as f:
            lines = f.readlines()

        data = np.genfromtxt(lines, dtype=np.float64)
        times = [l.split()[0] for l in lines]
        
        assert len(times) == len(data)

        pose_map = {}
        cam0_map = {}
        cam1_map = {}
        depth_map = {}
        
        for i in range(len(data)):
            
            ts = torch.tensor(data[i, 1:4])
            qs = torch.tensor(data[i, 4:])
            rs = torch.eye(4)
            rs[:3, :3] = torch.tensor(Rotation.from_quat(qs).as_matrix())
            rs[:3, 3] = ts
            pose = rs.to(torch.float32)

            pose_map[times[i]] = pose
            cam0_map[times[i]] = cam0_images_path / f"{times[i]}.png"
            cam1_map[times[i]] = cam1_images_path / f"{times[i]}.png"
            depth_map[times[i]] = keypoint_path / f"{times[i]}.txt"
        
        times.sort()
        """
        debug_path = self.dataset_dir / f"{seq}/monorec_data"
        print("Generating keyframes...")
        if os.path.isdir(debug_path):
            shutil.rmtree(debug_path)
        os.makedirs(debug_path / "keyframes/all/")
        with open(debug_path / "time_index_mapping.txt", 'w') as f: 
            for i, tns in enumerate(times): 
                shutil.copy2(cam0_map[tns], debug_path / f"keyframes/all/{i}.png")
                f.write(f'{i}\t{tns}\t{cam0_map[tns]}\t{pose_map[tns]}\n')
        print(f"[+]Generated data at {debug_path} for sequence {seq}")
        """
        return times, pose_map, cam0_map, cam1_map, depth_map

    # load pcalib
    def invert_pcalib(self, pcalib_path):
        assert os.path.exists(pcalib_path)
        pcalib = np.loadtxt(pcalib_path)
        inv_pcalib = torch.zeros(256, dtype=torch.float32)
        j = 0
        for i in range(256):
            while j < 255 and i + .5 > pcalib[j]:
                j += 1
            inv_pcalib[i] = j
        return inv_pcalib

    def open_image(self, index, seq, crop_box = None, keyframe_index = None, stereo = False):
        
        assert os.path.exists(self.cam0_paths[seq][self.pose_times[seq][index]])
        assert os.path.exists(self.cam1_paths[seq][self.pose_times[seq][index]])

        if stereo:
            img = Image.open(self.cam1_paths[seq][self.pose_times[seq][index]])
        else:
            img = Image.open(self.cam0_paths[seq][self.pose_times[seq][index]])
        img = img.convert('RGB')

        if crop_box:
            img = img.crop(crop_box)

        if self.resz_shape[seq]:
            img = img.resize((self.resz_shape[seq][1], self.resz_shape[seq][0]), resample=Image.BILINEAR)

        if keyframe_index and self.debug:
            debug_path = self.dataset_dir / f"{seq}/monorec_data"
            kf_dir = debug_path / f"keyframes/{keyframe_index}"
            if(not os.path.isdir(kf_dir)):
                os.mkdir(kf_dir)
            img.save(f'{kf_dir}/{index}.png')

        image_tensor = torch.tensor(np.array(img)).to(dtype=torch.float32)

        if stereo:
            image_tensor = self.cam1_pcalib[seq][image_tensor.to(dtype=torch.long)]
        else:
            image_tensor = self.cam0_pcalib[seq][image_tensor.to(dtype=torch.long)]
        
        image_tensor = image_tensor / 255 - .5
        if len(image_tensor.shape) == 2:
            image_tensor = torch.stack((image_tensor, image_tensor, image_tensor))
        else:
            image_tensor = image_tensor.permute(2, 0, 1)

        assert image_tensor.shape[1] == self.target_image_size[0]
        assert image_tensor.shape[2] == self.target_image_size[1]

        return image_tensor

        # for rectification check: 
        # https://github.com/tum-vision/mono_dataset_code/blob/master/src/BenchmarkDatasetReader.h
    
    def open_depth(self, index, seq, crop_box = None):
        
        depth = torch.zeros(1, self.orig_image_size[seq][0], self.orig_image_size[seq][1], dtype=torch.float32)
        assert os.path.exists(self.depth_paths[seq][self.pose_times[seq][index]])

        with open(self.depth_paths[seq][self.pose_times[seq][index]], "r") as f:
            lines = f.readlines()
        
        num_points = int(lines[0])
        if num_points > 0:
            for l in lines[1:]:
                tmp = l.split()
                x, y, i_depth = round(float(tmp[0])), round(float(tmp[1])), float(tmp[2])
                depth[0][y][x] = i_depth
        
        if crop_box:
            depth_cropped = depth[:,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]]
        else:
            depth_cropped = depth

        assert depth_cropped.shape[1] == self.target_image_size[0]
        assert depth_cropped.shape[2] == self.target_image_size[1]

        return depth_cropped

def format_intrinsics(cam0_intrinsics, cam1_intrinsics, target_image_size, orig_image_size):

    cam0_cx = cam0_intrinsics[0,2].item()
    cam0_cy = cam0_intrinsics[1,2].item()

    cam1_cx = cam1_intrinsics[0,2].item()
    cam1_cy = cam1_intrinsics[1,2].item()

    (new_h, new_w) = target_image_size
    (orig_h, orig_w) = orig_image_size

    assert new_h < orig_h
    assert new_w == orig_w

    cam0_cx_new = (cam0_cx + 0.5) - 0.5 - (orig_w - new_w)//2
    cam0_cy_new = (cam0_cy + 0.5) - 0.5 - (orig_h - new_h)//2

    cam1_cx_new = (cam1_cx + 0.5) - 0.5 - (orig_w - new_w)//2
    cam1_cy_new = (cam1_cy + 0.5) - 0.5 - (orig_h - new_h)//2

    cam0_intrinsics_new = cam0_intrinsics.clone()
    cam1_intrinsics_new = cam1_intrinsics.clone()

    cam0_intrinsics_new[0,2] = cam0_cx_new
    cam0_intrinsics_new[1,2] = cam0_cy_new

    cam1_intrinsics_new[0,2] = cam1_cx_new
    cam1_intrinsics_new[1,2] = cam1_cy_new
    
    box = ((orig_w - new_w)//2, (orig_h - new_h)//2, (orig_w - new_w)//2 + new_w, (orig_h - new_h)//2 + new_h)

    print(f"Orig Intrinsics: {cam0_intrinsics}")
    print(f"New Intrinsics: {cam0_intrinsics_new}")
    print(f"new_w: {new_w}")
    print(f"new_h: {new_h}")
    print(box)

    return cam0_intrinsics_new, cam1_intrinsics_new, None, box