import yaml
import io
import json
import argparse
from pathlib import Path
import numpy as np
import os
import shutil
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

"""
Usage:
python3 rectify.py -fov 0.25 -rp dataset-outdoors1_1024_16_rectified -bp ../basalt ../../tumvi_data/dataset-outdoors1_1024_16/
"""
class Rectifier():

    def __init__(self):

        # Initialize parser
        desc = "Python code for rectification of TUM-VI sequences. \nUpdated camera intrinsics and extrensics are written to a json file for basalt."
        parser = argparse.ArgumentParser(description = desc)

        # add arguments
        parser.add_argument("dataset_path", help="path to tum-vi sequence", type=str)
        parser.add_argument("-fov", "--field-of-view-scale", help="field of view scale for the new rectified image", type=float)
        parser.add_argument("-sc", "--new-scale", help="Scale for the new image size. New image size will be scale times the orig image size.", type=float)
        parser.add_argument("-rp", "--rectified-dirname", help="Directory name of the rectified dataset", type=str)
        parser.add_argument("-ip", "--intrinsic-path", help="Path for saving the intrinsics file (.yaml). Used only when intrinsics need to be saved without rectifying the entire dataset.", type=str)
        parser.add_argument("-bp", "--basalt-path", help="Path to basalt directory. Use to update rectified camera parameters in basalt.", type = str)
        parser.add_argument("-bcn", "--basalt-calib-name", help="Name of the calib file for basalt.", type = str)
        
        self.args = parser.parse_args()

        self.dataset_dir = Path(self.args.dataset_path)
        
        if self.args.field_of_view_scale:
            self.fov = self.args.field_of_view_scale
        else:
            self.fov = 1.0

        if self.args.new_scale:
            self.scale = self.args.new_scale
        else:
            self.scale = 1.0


        self.init_paths()
        self.load_yaml()
        self.calc_new_intrinsics()

        if self.args.rectified_dirname:
            self.rectify(self.args.rectified_dirname)

        elif self.args.intrinsic_path:
            self.save_new_intrinsics(self.args.intrinsic_path)

        if self.args.basalt_path and self.args.basalt_calib_name:
            self.store_basalt_calib(Path(self.args.basalt_path), self.args.basalt_calib_name)
        
    def init_paths(self):
        self.intrinsic_path = self.dataset_dir / "dso/camchain.yaml"
        self.cam0_path = self.dataset_dir / "mav0/cam0/data"
        self.cam1_path = self.dataset_dir / "mav0/cam1/data"
        
        assert os.path.exists(self.intrinsic_path)
        assert os.path.isdir(self.cam0_path)
        assert os.path.isdir(self.cam1_path)        

    def load_yaml(self):
        stream = open(self.intrinsic_path, "r")
        try:
            self.cam_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit()
        
        self.T_cam0_imu = np.array(self.cam_data["cam0"]["T_cam_imu"])
        self.T_cam1_imu = np.array(self.cam_data["cam1"]["T_cam_imu"])

        self.cam0_distortion = np.array(self.cam_data["cam0"]["distortion_coeffs"])
        self.cam1_distortion = np.array(self.cam_data["cam1"]["distortion_coeffs"])
        
        self.cam0_intrinsics = {
            "fx" : self.cam_data["cam0"]["intrinsics"][0],
            "fy" : self.cam_data["cam0"]["intrinsics"][1],
            "cx" : self.cam_data["cam0"]["intrinsics"][2],
            "cy" : self.cam_data["cam0"]["intrinsics"][3]
            }
        
        self.cam1_intrinsics = {
            "fx" : self.cam_data["cam1"]["intrinsics"][0],
            "fy" : self.cam_data["cam1"]["intrinsics"][1],
            "cx" : self.cam_data["cam1"]["intrinsics"][2],
            "cy" : self.cam_data["cam1"]["intrinsics"][3]
            }

        self.T_cam1_cam0 = np.array(self.cam_data["cam1"]["T_cn_cnm1"])
        self.R_cam1_cam0 = self.T_cam1_cam0[:3, :3]
        self.tvec_cam1_cam0 = self.T_cam1_cam0[:3, 3]

        self.cam0_intrinsic_mx = np.eye(3)
        self.cam0_intrinsic_mx[0,0] = self.cam0_intrinsics["fx"]
        self.cam0_intrinsic_mx[1,1] = self.cam0_intrinsics["fy"]
        self.cam0_intrinsic_mx[0,2] = self.cam0_intrinsics["cx"]
        self.cam0_intrinsic_mx[1,2] = self.cam0_intrinsics["cy"]

        self.cam1_intrinsic_mx = np.eye(3)
        self.cam1_intrinsic_mx[0,0] = self.cam1_intrinsics["fx"]
        self.cam1_intrinsic_mx[1,1] = self.cam1_intrinsics["fy"]
        self.cam1_intrinsic_mx[0,2] = self.cam1_intrinsics["cx"]
        self.cam1_intrinsic_mx[1,2] = self.cam1_intrinsics["cy"]

    def calc_new_intrinsics(self):
        self.cam0_images = os.listdir(self.cam0_path)
        self.cam1_images = os.listdir(self.cam1_path)

        self.cam0_images.sort()
        self.cam1_images.sort()

        # image size
        dummy = cv2.imread(str(self.cam0_path / self.cam0_images[0]), cv2.IMREAD_GRAYSCALE)
        h, w = dummy.shape
        self.imgSize = (h, w)
        
        # params
        self.new_imgSize = (int(self.scale * h), int(self.scale * w))
        flags = 0

        # projection matrix format:
        # [fx, 0,  cx, fx.tx]
        # [0,  fy, cy, 0]
        # [0,  0,  1,  0]

        # rectification
        self.R_rect0_cam0, self.R_rect1_cam1, self.P_rect0, self.P_rect1, Q = cv2.fisheye.stereoRectify(
            K1 = self.cam0_intrinsic_mx, 
            D1 = self.cam0_distortion, 
            K2 = self.cam1_intrinsic_mx, 
            D2 = self.cam1_distortion, 
            imageSize = self.imgSize, 
            R = self.R_cam1_cam0, 
            tvec = self.tvec_cam1_cam0, 
            flags=flags, 
            newImageSize=self.new_imgSize, 
            fov_scale = self.fov)

        self.rect0_intrinsics = {
            "fx" : self.P_rect0[0,0].item(),
            "fy" : self.P_rect0[1,1].item(),
            "cx" : self.P_rect0[0,2].item(),
            "cy" : self.P_rect0[1,2].item()
            }
        
        self.rect1_intrinsics = {
            "fx" : self.P_rect1[0,0].item(),
            "fy" : self.P_rect1[1,1].item(),
            "cx" : self.P_rect1[0,2].item(),
            "cy" : self.P_rect1[1,2].item()
            }
        
        # check
        # cv.fisheye.undistortImage(distorted, K, D[, undistorted[, Knew[, new_size]]])
        self.cam0_map1, self.cam0_map2 = cv2.fisheye.initUndistortRectifyMap(
            K = self.cam0_intrinsic_mx, 
            D = self.cam0_distortion, 
            R = self.R_rect0_cam0, 
            P = self.P_rect0, 
            size = self.new_imgSize, 
            m1type = cv2.CV_32FC1)

        self.cam1_map1, self.cam1_map2 = cv2.fisheye.initUndistortRectifyMap(
            K = self.cam1_intrinsic_mx, 
            D = self.cam1_distortion, 
            R = self.R_rect1_cam1, 
            P = self.P_rect1, 
            size = self.new_imgSize, 
            m1type = cv2.CV_32FC1)

        self.T_rect0_cam0 = np.zeros((4,4))
        self.T_rect0_cam0[3, 3] = 1.0
        self.T_rect0_cam0[:3, :3] = self.R_rect0_cam0

        self.T_rect1_cam1 = np.zeros((4,4))
        self.T_rect1_cam1[3, 3] = 1.0
        self.T_rect1_cam1[:3, :3] = self.R_rect1_cam1

        self.T_rect0_imu = self.T_rect0_cam0 @ self.T_cam0_imu
        self.T_rect1_imu = self.T_rect1_cam1 @ self.T_cam1_imu   

        self.T_rect1_rect0 = self.T_rect1_imu @ np.linalg.inv(self.T_rect0_imu)
        print("[+] Calculated new intrinsics. ")

    def rectify(self, rect_dirname):
        
        self.rect_path = Path(os.path.dirname(self.dataset_dir)) / f"{rect_dirname}"
        self.rect0_path = self.rect_path / "mav0/cam0/data"
        self.rect1_path = self.rect_path / "mav0/cam1/data"

        # create directories
        if(os.path.isdir(self.rect_path)):
            shutil.rmtree(self.rect_path)

        print(f"Creating directory structure for rectified TUM-VI sequence at {str(self.rect_path)}")
        os.mkdir(self.rect_path)
        os.makedirs(self.rect_path / "mav0/cam0/data")
        os.makedirs(self.rect_path / "mav0/cam1/data")

        shutil.copy(self.dataset_dir / "mav0/cam0/data.csv", self.rect_path / "mav0/cam0")
        shutil.copy(self.dataset_dir / "mav0/cam0/data.csv", self.rect_path / "mav0/cam0")
        shutil.copytree(self.dataset_dir / "mav0/imu0", self.rect_path / "mav0/imu0")
        shutil.copytree(self.dataset_dir / "mav0/mocap0", self.rect_path / "mav0/mocap0")

        os.makedirs(self.rect_path / "dso/cam0")
        os.makedirs(self.rect_path / "dso/cam1")

        shutil.copy(self.dataset_dir / "dso/gt_imu.csv", self.rect_path / "dso")
        shutil.copy(self.dataset_dir / "dso/imu.txt", self.rect_path / "dso")
        shutil.copy(self.dataset_dir / "dso/imu_config.yaml", self.rect_path / "dso")

        for file_name in os.listdir(self.dataset_dir / "dso/cam0"):
            source = self.dataset_dir / "dso/cam0" / file_name
            destination = self.rect_path / "dso/cam0"
            if os.path.isfile(source):
                shutil.copy(source, destination)

        for file_name in os.listdir(self.dataset_dir / "dso/cam1"):
            source = self.dataset_dir / "dso/cam1" / file_name
            destination = self.rect_path / "dso/cam1"
            if os.path.isfile(source):
                shutil.copy(source, destination)

        print(f"[+] Directory formation complete.")
        
        print("Rectifying cam0 images:")
        for img_name in tqdm(self.cam0_images):
            img = cv2.imread(str(self.cam0_path / img_name), cv2.IMREAD_GRAYSCALE)
            rect = cv2.remap(img, self.cam0_map1, self.cam0_map2, interpolation = cv2.INTER_LINEAR)
            cv2.imwrite(str(self.rect0_path / img_name), rect)

        print("Rectifying cam1 images:")
        for img_name in tqdm(self.cam1_images):
            img = cv2.imread(str(self.cam1_path / img_name), cv2.IMREAD_GRAYSCALE)
            rect = cv2.remap(img, self.cam1_map1, self.cam1_map2, interpolation = cv2.INTER_LINEAR)
            cv2.imwrite(str(self.rect1_path / img_name), rect)

        os.symlink(os.path.abspath(self.rect_path / "mav0/cam0/data"), self.rect_path / "dso/cam0/images")
        os.symlink(os.path.abspath(self.rect_path / "mav0/cam1/data"), self.rect_path / "dso/cam1/images")

        print(f"[+] Rectification of dataset sequence complete.")
        self.save_new_intrinsics(self.rect_path / "dso/camchain.yaml")

    def save_new_intrinsics(self, new_intrinsic_path):
                
        rectified_cam_data = self.cam_data.copy()
        rectified_cam_data["cam0"]["T_cam_imu"] = self.T_rect0_imu.tolist()
        rectified_cam_data["cam1"]["T_cam_imu"] = self.T_rect1_imu.tolist()

        rectified_cam_data["cam1"]["T_cn_cnm1"] = self.T_rect1_rect0.tolist()

        rectified_cam_data["cam0"]["distortion_coeffs"] = [0.0, 0.0, 0.0, 0.0]
        rectified_cam_data["cam1"]["distortion_coeffs"] = [0.0, 0.0, 0.0, 0.0]

        rectified_cam_data["cam0"]["intrinsics"] = [
            self.rect0_intrinsics["fx"],
            self.rect0_intrinsics["fy"],
            self.rect0_intrinsics["cx"],
            self.rect0_intrinsics["cy"]
        ]
        rectified_cam_data["cam1"]["intrinsics"] = [
            self.rect1_intrinsics["fx"],
            self.rect1_intrinsics["fy"],
            self.rect1_intrinsics["cx"],
            self.rect1_intrinsics["cy"]
        ]
        # print(type(rectified_cam_data["cam0"]["intrinsics"][0].item()))

        rectified_cam_data["cam0"]["resolution"] = [self.new_imgSize[0], self.new_imgSize[1]]
        rectified_cam_data["cam1"]["resolution"] = [self.new_imgSize[0], self.new_imgSize[1]]

        with open(new_intrinsic_path, 'w') as outfile:
            yaml.safe_dump(rectified_cam_data, outfile, default_flow_style=None, allow_unicode=False)
        print(f"[+] New intrinsics saved and updated at {new_intrinsic_path}.")
    
    def store_basalt_calib(self, basalt_dir, calib_name):
        calib_path = basalt_dir / "data/tumvi_512_ds_calib.json"
        new_calib_path = basalt_dir / f"data/{calib_name}.json"
        
        assert os.path.exists(calib_path)

        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        
        T_imu_rect0 = np.linalg.inv(self.T_rect0_imu)
        T_imu_rect1 = np.linalg.inv(self.T_rect1_imu)
        
        r_imu_rect0 = R.from_matrix(T_imu_rect0[:3,:3])
        Q_imu_rect0 = r_imu_rect0.as_quat()

        r_imu_rect1 = R.from_matrix(T_imu_rect1[:3,:3])
        Q_imu_rect1 = r_imu_rect1.as_quat()

        new_calib_data = calib_data.copy()
        new_calib_data["value0"]["T_imu_cam"][0]["px"] = T_imu_rect0[0,3] 
        new_calib_data["value0"]["T_imu_cam"][0]["py"] = T_imu_rect0[1,3] 
        new_calib_data["value0"]["T_imu_cam"][0]["pz"] = T_imu_rect0[2,3] 
        new_calib_data["value0"]["T_imu_cam"][0]["qx"] = Q_imu_rect0[0] 
        new_calib_data["value0"]["T_imu_cam"][0]["qy"] = Q_imu_rect0[1] 
        new_calib_data["value0"]["T_imu_cam"][0]["qz"] = Q_imu_rect0[2] 
        new_calib_data["value0"]["T_imu_cam"][0]["qw"] = Q_imu_rect0[3] 

        new_calib_data["value0"]["T_imu_cam"][1]["px"] = T_imu_rect1[0,3] 
        new_calib_data["value0"]["T_imu_cam"][1]["py"] = T_imu_rect1[1,3] 
        new_calib_data["value0"]["T_imu_cam"][1]["pz"] = T_imu_rect1[2,3] 
        new_calib_data["value0"]["T_imu_cam"][1]["qx"] = Q_imu_rect1[0] 
        new_calib_data["value0"]["T_imu_cam"][1]["qy"] = Q_imu_rect1[1] 
        new_calib_data["value0"]["T_imu_cam"][1]["qz"] = Q_imu_rect1[2] 
        new_calib_data["value0"]["T_imu_cam"][1]["qw"] = Q_imu_rect1[3] 

        new_calib_data["value0"]["intrinsics"][0]["camera_type"] = "pinhole" 
        new_calib_data["value0"]["intrinsics"][1]["camera_type"] = "pinhole" 

        new_calib_data["value0"]["intrinsics"][0]["intrinsics"].pop("xi") 
        new_calib_data["value0"]["intrinsics"][0]["intrinsics"].pop("alpha") 
        new_calib_data["value0"]["intrinsics"][1]["intrinsics"].pop("xi") 
        new_calib_data["value0"]["intrinsics"][1]["intrinsics"].pop("alpha") 

        new_calib_data["value0"]["intrinsics"][0]["intrinsics"]["fx"] = self.rect0_intrinsics["fx"] 
        new_calib_data["value0"]["intrinsics"][0]["intrinsics"]["fy"] = self.rect0_intrinsics["fy"] 
        new_calib_data["value0"]["intrinsics"][0]["intrinsics"]["cx"] = self.rect0_intrinsics["cx"] 
        new_calib_data["value0"]["intrinsics"][0]["intrinsics"]["cy"] = self.rect0_intrinsics["cy"] 

        new_calib_data["value0"]["intrinsics"][1]["intrinsics"]["fx"] = self.rect1_intrinsics["fx"] 
        new_calib_data["value0"]["intrinsics"][1]["intrinsics"]["fy"] = self.rect1_intrinsics["fy"] 
        new_calib_data["value0"]["intrinsics"][1]["intrinsics"]["cx"] = self.rect1_intrinsics["cx"] 
        new_calib_data["value0"]["intrinsics"][1]["intrinsics"]["cy"] = self.rect1_intrinsics["cy"] 

        new_calib_data["value0"]["resolution"][0] =  [self.new_imgSize[0], self.new_imgSize[1]]
        new_calib_data["value0"]["resolution"][1] =  [self.new_imgSize[0], self.new_imgSize[1]]

        with open(new_calib_path, 'w') as json_file:
            json.dump(new_calib_data, json_file, indent = 4)
        print(f"[+] New intrinsics updated in basalt code at {new_calib_path}.")

if __name__=="__main__":
    main = Rectifier()
