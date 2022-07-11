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
python3 prep_basalt.py ../../tumvi_data/dataset-outdoors1_512_16_rectified ../basalt test
"""
class Basalt_prepper():

    def __init__(self):

        # Initialize parser
        desc = "Python code for using TUM-VI dataset to update camera intrinsics and extrensics for basalt."
        parser = argparse.ArgumentParser(description = desc)

        # add arguments
        parser.add_argument("tumvi_path", help="path to tum-vi sequence", type=str)
        parser.add_argument("basalt_path", help="path to basalt directory", type=str)
        parser.add_argument("calib_name", help="name of new calib file for basalt", type=str)
               
        self.args = parser.parse_args()
        self.tumvi_dir = Path(self.args.tumvi_path)
        self.basalt_dir = Path(self.args.basalt_path)
        self.calib_name = self.args.calib_name

        self.init_paths()
        self.load_yaml()
        self.store_basalt_calib()
        
    def init_paths(self):
        self.intrinsic_path = self.tumvi_dir / "dso/camchain.yaml"
        assert os.path.exists(self.intrinsic_path)
 
        self.calib_path = self.basalt_dir / "data/tumvi_512_ds_calib.json"
        self.new_calib_path = self.basalt_dir / f"data/{self.calib_name}.json"
        
        assert os.path.exists(self.calib_path)

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
    
    def store_basalt_calib(self):

        with open(self.calib_path, 'r') as f:
            calib_data = json.load(f)
        
        T_imu_rect0 = np.linalg.inv(self.T_cam0_imu)
        T_imu_rect1 = np.linalg.inv(self.T_cam1_imu)
        
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

        new_calib_data["value0"]["intrinsics"][0]["intrinsics"]["fx"] = self.cam0_intrinsics["fx"] 
        new_calib_data["value0"]["intrinsics"][0]["intrinsics"]["fy"] = self.cam0_intrinsics["fy"] 
        new_calib_data["value0"]["intrinsics"][0]["intrinsics"]["cx"] = self.cam0_intrinsics["cx"] 
        new_calib_data["value0"]["intrinsics"][0]["intrinsics"]["cy"] = self.cam0_intrinsics["cy"] 

        new_calib_data["value0"]["intrinsics"][1]["intrinsics"]["fx"] = self.cam1_intrinsics["fx"] 
        new_calib_data["value0"]["intrinsics"][1]["intrinsics"]["fy"] = self.cam1_intrinsics["fy"] 
        new_calib_data["value0"]["intrinsics"][1]["intrinsics"]["cx"] = self.cam1_intrinsics["cx"] 
        new_calib_data["value0"]["intrinsics"][1]["intrinsics"]["cy"] = self.cam1_intrinsics["cy"] 

        new_calib_data["value0"]["resolution"][0] =  self.cam_data["cam0"]["resolution"]
        new_calib_data["value0"]["resolution"][1] =  self.cam_data["cam1"]["resolution"]

        with open(self.new_calib_path, 'w') as json_file:
            json.dump(new_calib_data, json_file, indent = 4)

if __name__=="__main__":
    main = Basalt_prepper()
