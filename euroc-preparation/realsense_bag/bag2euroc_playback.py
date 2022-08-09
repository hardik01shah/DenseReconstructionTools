#! /usr/bin/env python3
import rospy
import numpy as np
import argparse
from pathlib import Path
import os
import shutil
import csv
import yaml
import json

from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import Transform
import cv_bridge
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

from math import modf

"""
usage:
1. roscore
2. rosbag play -d 10 test.bag
3. python3 bag2euroc.py ./euroc/
"""
class Bag2Euroc:

    bridge = cv_bridge.CvBridge()

    cam0_tf = None
    cam1_tf = None
    cam0_info = None
    cam1_info = None
    accel_tf = None
    gyro_tf = None

    gyro_data = {}
    accel_data = {}

    def __init__(self, euroc_dir):

        self.euroc_dir = Path(os.path.abspath(euroc_dir))
        self.init_paths()

        print(f"Initializing Subscibers...")

        self.cam0_data_sub = rospy.Subscriber('/device_0/sensor_0/Infrared_1/image/data', Image, self.cam0_data_callback)
        self.cam1_data_sub = rospy.Subscriber('/device_0/sensor_0/Infrared_2/image/data', Image, self.cam1_data_callback)
        self.cam0_info_sub = rospy.Subscriber('/device_0/sensor_0/Infrared_1/info/camera_info', CameraInfo, self.cam0_info_callback)
        self.cam1_info_sub = rospy.Subscriber('/device_0/sensor_0/Infrared_2/info/camera_info', CameraInfo, self.cam1_info_callback)
        self.cam0_tf_sub = rospy.Subscriber('/device_0/sensor_0/Infrared_1/tf/0', Transform, self.cam0_tf_callback)
        self.cam1_tf_sub = rospy.Subscriber('/device_0/sensor_0/Infrared_2/tf/0', Transform, self.cam1_tf_callback)
        
        self.rgb_data_sub = rospy.Subscriber('/device_0/sensor_1/Color_0/image/data', Image, self.rgb_data_callback)

        self.accel_data_sub = rospy.Subscriber('/device_0/sensor_2/Accel_0/imu/data', Imu, self.accel_data_callback)
        self.accel_tf_sub = rospy.Subscriber('/device_0/sensor_2/Accel_0/tf/0', Transform, self.accel_tf_callback)

        self.gyro_data_sub = rospy.Subscriber('/device_0/sensor_2/Gyro_0/imu/data', Imu, self.gyro_data_callback)
        self.gyro_tf_sub = rospy.Subscriber('/device_0/sensor_2/Gyro_0/tf/0', Transform, self.gyro_tf_callback)
        
        print(f"[+] Subscibers initialized. Waiting for topics...")

        while not(self.cam0_info!=None and self.cam1_info!=None) and not(rospy.is_shutdown()):
            continue
        assert self.cam0_info == self.cam1_info
        print(f"[+] Recieved information for both cameras.")

        while not(self.cam0_tf!=None and self.cam1_tf!=None and self.gyro_tf!=None and self.accel_tf!=None) and not(rospy.is_shutdown()):
            continue
        assert self.accel_tf == self.gyro_tf
        print(f"[+] Recieved transforms for both cameras and imu.")

        # wait for cam topics to stop publishing
        cam0_topic = ['/device_0/sensor_0/Infrared_1/image/data', 'sensor_msgs/Image']
        cam1_topic = ['/device_0/sensor_0/Infrared_2/image/data', 'sensor_msgs/Image']
        while ((rospy.get_published_topics().count(cam0_topic) > 0) or (rospy.get_published_topics().count(cam1_topic) > 0)) and not(rospy.is_shutdown()):
            continue

        self.f0.close()
        self.f1.close()        
        print(f"[+] Saved camera0 images at {self.cam0_path} and information at {self.cam0_csv_path}.")
        print(f"[+] Saved camera1 images at {self.cam1_path} and information at {self.cam1_csv_path}.")

        # wait for imu topics to stop publishing
        accel_topic = ['/device_0/sensor_2/Accel_0/imu/data', 'sensor_msgs/Imu']
        gyro_topic = ['/device_0/sensor_2/Gyro_0/imu/data', 'sensor_msgs/Imu']
        while ((rospy.get_published_topics().count(accel_topic) > 0) or (rospy.get_published_topics().count(gyro_topic) > 0)) and not(rospy.is_shutdown()):
            continue
        
        self.save_imu_data()
        self.save_imu_info()
        self.save_cam_info()
        self.save_basalt_calib()

        print("[+] Done.")
        exit()

    def init_paths(self):
        self.cam0_path = self.euroc_dir / "mav0/cam0/data"
        self.cam0_csv_path = self.euroc_dir / "mav0/cam0/data.csv"
        self.cam1_path = self.euroc_dir / "mav0/cam1/data"
        self.cam1_csv_path = self.euroc_dir / "mav0/cam1/data.csv"
        self.imu_path = self.euroc_dir / "mav0/imu0"
        self.imu_csv_path = self.euroc_dir / "mav0/imu0/data.csv"
        self.imu_txt_path = self.euroc_dir / "dso/imu.txt"
        self.mocap_path = self.euroc_dir / "mav0/mocap0"

        self.rgb_path = self.euroc_dir / "mav0/rgb/data"

        self.camchain_path = self.euroc_dir / "dso/camchain.yaml"
        self.imu_info_path = self.euroc_dir / "dso/imu_config.yaml"

        self.basalt_calib_path = self.euroc_dir / "basalt_calib.json"

        # create directories
        if(os.path.isdir(self.euroc_dir)):
            shutil.rmtree(self.euroc_dir)

        print(f"Creating directory structure for Euroc dataset at {str(self.euroc_dir)}")
        os.mkdir(self.euroc_dir)
        os.makedirs(self.cam0_path)
        os.makedirs(self.cam1_path)
        os.makedirs(self.rgb_path)
        os.makedirs(self.imu_path)
        os.makedirs(self.mocap_path)

        cam_csv_header = ['# timestamp [ns]', 'filename']
        self.f0 = open(self.cam0_csv_path, 'w')
        self.cam0_csv_writer = csv.writer(self.f0)
        self.cam0_csv_writer.writerow(cam_csv_header)

        self.f1 = open(self.cam1_csv_path, 'w')
        self.cam1_csv_writer = csv.writer(self.f1)
        self.cam1_csv_writer.writerow(cam_csv_header)

        print(f"[+] Directory created.")

    def save_imu_data(self):

        self.dso_cam0_dir = self.euroc_dir / f"dso/cam0"
        self.dso_cam1_dir = self.euroc_dir / f"dso/cam1"

        os.makedirs(self.dso_cam0_dir)
        os.makedirs(self.dso_cam1_dir)
        os.symlink(os.path.abspath(self.cam0_path), self.dso_cam0_dir / "images")
        os.symlink(os.path.abspath(self.cam1_path), self.dso_cam1_dir / "images")

        imu_csv_header = ['#timestamp [ns]', 'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]', 'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']
        self.fi = open(self.imu_csv_path, 'w')
        self.imu_csv_writer = csv.DictWriter(self.fi, imu_csv_header)
        self.imu_csv_writer.writeheader()
        self.ft = open(self.imu_txt_path, 'w')
        imu_txt_header = "# timestamp[ns] w.x w.y w.z a.x a.y a.z"
        self.ft.write(f"{imu_txt_header}\n")
        # for ts in self.gyro_data.keys():
        #     if ts in self.accel_data.keys():
        #         row = {
        #             '#timestamp [ns]': ts, 
        #             'w_RS_S_x [rad s^-1]': self.gyro_data[ts]['w_RS_S_x [rad s^-1]'],
        #             'w_RS_S_y [rad s^-1]': self.gyro_data[ts]['w_RS_S_y [rad s^-1]'],
        #             'w_RS_S_z [rad s^-1]': self.gyro_data[ts]['w_RS_S_z [rad s^-1]'],
        #             'a_RS_S_x [m s^-2]': self.accel_data[ts]['a_RS_S_x [m s^-2]'],
        #             'a_RS_S_y [m s^-2]': self.accel_data[ts]['a_RS_S_y [m s^-2]'],
        #             'a_RS_S_z [m s^-2]': self.accel_data[ts]['a_RS_S_z [m s^-2]']
        #         }
        #         self.imu_csv_writer.writerow(row)
        #         self.ft.write(f"{ts} {self.gyro_data[ts]['w_RS_S_x [rad s^-1]']} {self.gyro_data[ts]['w_RS_S_y [rad s^-1]']} {self.gyro_data[ts]['w_RS_S_z [rad s^-1]']} {self.accel_data[ts]['a_RS_S_x [m s^-2]']} {self.accel_data[ts]['a_RS_S_y [m s^-2]']} {self.accel_data[ts]['a_RS_S_z [m s^-2]']}\n")
        #     else:
        #         row = {
        #             '#timestamp [ns]': ts, 
        #             'w_RS_S_x [rad s^-1]': self.gyro_data[ts]['w_RS_S_x [rad s^-1]'],
        #             'w_RS_S_y [rad s^-1]': self.gyro_data[ts]['w_RS_S_y [rad s^-1]'],
        #             'w_RS_S_z [rad s^-1]': self.gyro_data[ts]['w_RS_S_z [rad s^-1]'],
        #             'a_RS_S_x [m s^-2]': 0.0,
        #             'a_RS_S_y [m s^-2]': 0.0,
        #             'a_RS_S_z [m s^-2]': 0.0
        #         }
        #         self.imu_csv_writer.writerow(row)
        
        # for ts in self.accel_data.keys():
        #     if ts not in self.gyro_data.keys():
        #         row = {
        #             '#timestamp [ns]': ts, 
        #             'w_RS_S_x [rad s^-1]': 0.0,
        #             'w_RS_S_y [rad s^-1]': 0.0,
        #             'w_RS_S_z [rad s^-1]': 0.0,
        #             'a_RS_S_x [m s^-2]': self.accel_data[ts]['a_RS_S_x [m s^-2]'],
        #             'a_RS_S_y [m s^-2]': self.accel_data[ts]['a_RS_S_y [m s^-2]'],
        #             'a_RS_S_z [m s^-2]': self.accel_data[ts]['a_RS_S_z [m s^-2]']
        #         }
        #         self.imu_csv_writer.writerow(row)

        gyro_ns = []
        gyro_map = {}
        for ts in self.gyro_data.keys():
            num = ts.secs+(ts.nsecs*1e-9)
            gyro_ns.append(num)
            gyro_map[num] = ts
        
        accel_ns = []
        accel_map = {}
        for ts in self.accel_data.keys():
            num = ts.secs+(ts.nsecs*1e-9)
            accel_ns.append(num)
            accel_map[num] = ts

        gyro_ns.sort()
        accel_ns.sort()
        np_accel_ns = np.array(accel_ns)

        for ns in gyro_ns:

            a_ns_2, a_ns_1 = self.closest(np_accel_ns, ns)
            if(not a_ns_1):
                continue
            if not(a_ns_1 <= ns <= a_ns_2):
                continue
            
            a_ts_1 = accel_map[a_ns_1]
            a_ts_2 = accel_map[a_ns_2]
            ts = gyro_map[ns]

            f_x = interp1d([a_ns_1, a_ns_2], [self.accel_data[a_ts_1]['a_RS_S_x [m s^-2]'], self.accel_data[a_ts_2]['a_RS_S_x [m s^-2]']])
            f_y = interp1d([a_ns_1, a_ns_2], [self.accel_data[a_ts_1]['a_RS_S_y [m s^-2]'], self.accel_data[a_ts_2]['a_RS_S_y [m s^-2]']])
            f_z = interp1d([a_ns_1, a_ns_2], [self.accel_data[a_ts_1]['a_RS_S_z [m s^-2]'], self.accel_data[a_ts_2]['a_RS_S_z [m s^-2]']])
            
            row = {
                '#timestamp [ns]': ts, 
                'w_RS_S_x [rad s^-1]': self.gyro_data[ts]['w_RS_S_x [rad s^-1]'],
                'w_RS_S_y [rad s^-1]': self.gyro_data[ts]['w_RS_S_y [rad s^-1]'],
                'w_RS_S_z [rad s^-1]': self.gyro_data[ts]['w_RS_S_z [rad s^-1]'],
                'a_RS_S_x [m s^-2]': f_x(ns).item(),
                'a_RS_S_y [m s^-2]': f_y(ns).item(),
                'a_RS_S_z [m s^-2]': f_z(ns).item()
            }
            self.imu_csv_writer.writerow(row)
            self.ft.write(f"{ts} {self.gyro_data[ts]['w_RS_S_x [rad s^-1]']} {self.gyro_data[ts]['w_RS_S_y [rad s^-1]']} {self.gyro_data[ts]['w_RS_S_z [rad s^-1]']} {f_x(ns).item()} {f_y(ns).item()} {f_z(ns).item()}\n")

        self.fi.close()
        self.ft.close()

        print(f"[+] Saved imu data at {self.imu_csv_path} and {self.imu_txt_path}.")

    def closest(self, lst, K):
        if lst[lst > K].shape[0]==0 or lst[lst < K].shape[0]==0:
            return None, None
        return lst[lst > K].min(), lst[lst < K].max()

    def save_imu_info(self):

        # this has been hardcoded as access to topics is denied.
        # message types realsense_msgs/ImuIntrinsic and realsense_msgs/StreamInfo are proprietory
        imu_config = {}
        imu_config["accelerometer_rostopic"] = "/device_0/sensor_2/Accel_0/imu/data"
        imu_config["accelerometer_update_rate"] = 200.0
        imu_config["gyroscope_rostopic"] = "/device_0/sensor_2/Gyro_0/imu/data"
        imu_config["gyroscope_update_rate"] = 200.0

        imu_config["accelerometer_noise_density"] = 0.0
        imu_config["accelerometer_random_walk"] = 0.0
        imu_config["gyroscope_noise_density"] = 0.0
        imu_config["gyroscope_random_walk"] = 0.0

        with open(self.imu_info_path, 'w') as outfile:
            yaml.dump(imu_config, outfile, default_flow_style=False, allow_unicode=True)
        print(f"[+] New imu intrinsics saved at {self.imu_info_path}.")

    def save_cam_info(self):

        camchain_data = {}
        camchain_data["cam0"] = {}
        camchain_data["cam1"] = {}

        T_w_cam0 = self.get_T_matrix(self.cam0_tf)
        T_w_cam1 = self.get_T_matrix(self.cam1_tf)
        T_w_imu = self.get_T_matrix(self.gyro_tf)

        # T_cam0_imu = inv(T_w_cam0) * T_w_imu
        self.T_cam0_imu = np.linalg.inv(T_w_cam0) @ T_w_imu

        # T_cam1_imu = inv(T_w_cam1) * T_w_imu
        self.T_cam1_imu = np.linalg.inv(T_w_cam1) @ T_w_imu

        # T_cam1_cam0 = inv(T_w_cam1) * T_w_cam0
        self.T_cam1_cam0 = np.linalg.inv(T_w_cam1) @ T_w_cam0
        
        camchain_data['cam0']['T_cam_imu'] = self.T_cam0_imu.tolist()
        camchain_data['cam1']['T_cam_imu'] = self.T_cam1_imu.tolist()
        camchain_data['cam1']['T_cn_cnm1'] = self.T_cam1_cam0.tolist()

        camchain_data['cam0']['cam_overlaps'] = [1]
        camchain_data['cam1']['cam_overlaps'] = [0]

        camchain_data['cam0']['camera_model'] = "pinhole"
        camchain_data['cam1']['camera_model'] = "pinhole"
    
        camchain_data['cam0']['distortion_coeffs'] = self.cam0_info.D
        camchain_data['cam1']['distortion_coeffs'] = self.cam1_info.D
        
        camchain_data['cam0']['distortion_model'] = self.cam0_info.distortion_model
        camchain_data['cam1']['distortion_model'] = self.cam1_info.distortion_model

        camchain_data['cam0']['intrinsics'] = [self.cam0_info.K[0], self.cam0_info.K[4], self.cam0_info.K[2], self.cam0_info.K[5]]
        camchain_data['cam1']['intrinsics'] = [self.cam1_info.K[0], self.cam1_info.K[4], self.cam1_info.K[2], self.cam1_info.K[5]]

        camchain_data['cam0']['resolution'] = [self.cam0_info.height, self.cam0_info.width]
        camchain_data['cam1']['resolution'] = [self.cam1_info.height, self.cam1_info.width]

        camchain_data['cam0']['rostopic'] = '/device_0/sensor_0/Infrared_1/image/data'
        camchain_data['cam1']['rostopic'] = '/device_0/sensor_0/Infrared_2/image/data'

        with open(self.camchain_path, 'w') as outfile:
            yaml.safe_dump(camchain_data, outfile, default_flow_style=None, allow_unicode=False)
        print(f"[+] New intrinsics saved at {self.camchain_path}.")

    def get_T_matrix(self, tf):
        T_f_w = np.eye(4)
        tvec_f_w = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
        q_f_w = [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]
        r_f_w = R.from_quat(q_f_w).as_matrix()

        T_f_w[:3,3] = tvec_f_w
        T_f_w[:3,:3] = r_f_w
        return np.linalg.inv(T_f_w)

    def save_basalt_calib(self):
        calib_data = {
            "value0": {
                "T_imu_cam": [
                    {
                        "px": 0.0,
                        "py": 0.0,
                        "pz": 0.0,
                        "qx": 0.0,
                        "qy": 0.0,
                        "qz": 0.0,
                        "qw": 1.0
                    },
                    {
                        "px": 0.0,
                        "py": 0.0,
                        "pz": 0.0,
                        "qx": 0.0,
                        "qy": 0.0,
                        "qz": 0.0,
                        "qw": 1.0
                    }
                ],
                "intrinsics": [
                    {
                        "camera_type": "pinhole",
                        "intrinsics": {
                            "fx": 0.0,
                            "fy": 0.0,
                            "cx": 0.0,
                            "cy": 0.0
                        }
                    },
                    {
                        "camera_type": "pinhole",
                        "intrinsics": {
                            "fx": 0.0,
                            "fy": 0.0,
                            "cx": 0.0,
                            "cy": 0.0
                        }
                    }
                ],
                "resolution": [
                    [
                        0,
                        0
                    ],
                    [
                        0,
                        0
                    ]
                ],
                "calib_accel_bias": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                "calib_gyro_bias": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                "imu_update_rate": 200.0,
                "accel_noise_std": [
                    0.0,
                    0.0,
                    0.0
                ],
                "gyro_noise_std": [
                    0.0,
                    0.0,
                    0.0
                ],
                "accel_bias_std": [
                    0.0,
                    0.0,
                    0.0
                ],
                "gyro_bias_std": [
                    0.0,
                    0.0,
                    0.0
                ],
                "T_mocap_world": {
                    "px": 0.0,
                    "py": 0.0,
                    "pz": 0.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0
                },
                "T_imu_marker": {
                    "px": 0.0,
                    "py": 0.0,
                    "pz": 0.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0
                },
                "mocap_time_offset_ns": 0,
                "mocap_to_imu_offset_ns": 0,
                "cam_time_offset_ns": 0,
                "vignette": [
                    {
                        "value0": 0,
                        "value1": 10000000000,
                        "value2": [
                            [
                                0.9899372881888493
                            ],
                            [
                                0.9931178795891834
                            ],
                            [
                                0.9984407803536122
                            ],
                            [
                                0.996964178110498
                            ],
                            [
                                0.9956605833036875
                            ],
                            [
                                0.9973489553595134
                            ],
                            [
                                0.9985691891331739
                            ],
                            [
                                1.0
                            ],
                            [
                                0.9966866746094087
                            ],
                            [
                                0.9958706368580655
                            ],
                            [
                                0.9957807702286541
                            ],
                            [
                                0.9927911027519979
                            ],
                            [
                                0.9838733297580375
                            ],
                            [
                                0.9719198145611155
                            ],
                            [
                                0.9566058020047663
                            ],
                            [
                                0.9378032130699772
                            ],
                            [
                                0.9194004905192558
                            ],
                            [
                                0.9024065312353705
                            ],
                            [
                                0.8845584821495792
                            ],
                            [
                                0.8624817202873822
                            ],
                            [
                                0.8449163722596605
                            ],
                            [
                                0.8239771250783265
                            ],
                            [
                                0.802936456716557
                            ],
                            [
                                0.7811632337206126
                            ],
                            [
                                0.758539495055499
                            ],
                            [
                                0.7341063700839284
                            ],
                            [
                                0.7100276770866111
                            ],
                            [
                                0.6874554816035845
                            ],
                            [
                                0.6641863547515791
                            ],
                            [
                                0.6412571843680321
                            ],
                            [
                                0.6228323769487529
                            ],
                            [
                                0.6030940673078892
                            ],
                            [
                                0.5598212055840929
                            ],
                            [
                                0.4377741720108287
                            ],
                            [
                                0.24026566347390074
                            ],
                            [
                                0.08222347033252997
                            ],
                            [
                                0.03489917830551274
                            ],
                            [
                                0.029153547460757365
                            ],
                            [
                                0.030477276736751078
                            ],
                            [
                                0.01
                            ],
                            [
                                0.01
                            ],
                            [
                                0.01
                            ],
                            [
                                0.01
                            ]
                        ]
                    },
                    {
                        "value0": 0,
                        "value1": 10000000000,
                        "value2": [
                            [
                                0.9982144512705363
                            ],
                            [
                                0.9961165266369948
                            ],
                            [
                                0.9892083034957775
                            ],
                            [
                                0.9870647327777502
                            ],
                            [
                                0.9863145153017516
                            ],
                            [
                                0.9847218081828388
                            ],
                            [
                                0.9828536767530198
                            ],
                            [
                                0.9808724309018916
                            ],
                            [
                                0.9797280962140972
                            ],
                            [
                                0.9784762156394231
                            ],
                            [
                                0.9771015659455632
                            ],
                            [
                                0.9730709135912151
                            ],
                            [
                                0.9650117828320545
                            ],
                            [
                                0.9512319166574508
                            ],
                            [
                                0.9346333455674182
                            ],
                            [
                                0.9182278021221385
                            ],
                            [
                                0.9025351121100457
                            ],
                            [
                                0.8845788263796353
                            ],
                            [
                                0.8668369304290567
                            ],
                            [
                                0.8469326512989164
                            ],
                            [
                                0.8280417962134851
                            ],
                            [
                                0.8072181976634909
                            ],
                            [
                                0.788496496511515
                            ],
                            [
                                0.7691952807935
                            ],
                            [
                                0.7459299901795546
                            ],
                            [
                                0.7240742868053557
                            ],
                            [
                                0.7009088004583106
                            ],
                            [
                                0.6799474150660547
                            ],
                            [
                                0.6586062723412073
                            ],
                            [
                                0.6406388063878976
                            ],
                            [
                                0.6239758367746359
                            ],
                            [
                                0.6026729560290718
                            ],
                            [
                                0.5455288113795859
                            ],
                            [
                                0.3945979030583888
                            ],
                            [
                                0.1981919324270445
                            ],
                            [
                                0.06754460426558173
                            ],
                            [
                                0.028910567836990936
                            ],
                            [
                                0.02842995279817033
                            ],
                            [
                                0.0299618927515648
                            ],
                            [
                                0.01
                            ],
                            [
                                0.01
                            ],
                            [
                                0.01
                            ],
                            [
                                0.01
                            ]
                        ]
                    }
                ]
            }
        }

        T_imu_cam0 = np.linalg.inv(self.T_cam0_imu)
        T_imu_cam1 = np.linalg.inv(self.T_cam1_imu)
        
        r_imu_cam0 = R.from_matrix(T_imu_cam0[:3,:3])
        Q_imu_cam0 = r_imu_cam0.as_quat()

        r_imu_cam1 = R.from_matrix(T_imu_cam1[:3,:3])
        Q_imu_cam1 = r_imu_cam1.as_quat()

        calib_data["value0"]["T_imu_cam"][0]["px"] = T_imu_cam0[0,3] 
        calib_data["value0"]["T_imu_cam"][0]["py"] = T_imu_cam0[1,3] 
        calib_data["value0"]["T_imu_cam"][0]["pz"] = T_imu_cam0[2,3] 
        calib_data["value0"]["T_imu_cam"][0]["qx"] = Q_imu_cam0[0] 
        calib_data["value0"]["T_imu_cam"][0]["qy"] = Q_imu_cam0[1] 
        calib_data["value0"]["T_imu_cam"][0]["qz"] = Q_imu_cam0[2] 
        calib_data["value0"]["T_imu_cam"][0]["qw"] = Q_imu_cam0[3] 

        calib_data["value0"]["T_imu_cam"][1]["px"] = T_imu_cam1[0,3] 
        calib_data["value0"]["T_imu_cam"][1]["py"] = T_imu_cam1[1,3] 
        calib_data["value0"]["T_imu_cam"][1]["pz"] = T_imu_cam1[2,3] 
        calib_data["value0"]["T_imu_cam"][1]["qx"] = Q_imu_cam1[0] 
        calib_data["value0"]["T_imu_cam"][1]["qy"] = Q_imu_cam1[1] 
        calib_data["value0"]["T_imu_cam"][1]["qz"] = Q_imu_cam1[2] 
        calib_data["value0"]["T_imu_cam"][1]["qw"] = Q_imu_cam1[3] 

        calib_data["value0"]["intrinsics"][0]["camera_type"] = "pinhole" 
        calib_data["value0"]["intrinsics"][1]["camera_type"] = "pinhole" 

        calib_data["value0"]["intrinsics"][0]["intrinsics"]["fx"] = self.cam0_info.K[0] 
        calib_data["value0"]["intrinsics"][0]["intrinsics"]["fy"] = self.cam0_info.K[4] 
        calib_data["value0"]["intrinsics"][0]["intrinsics"]["cx"] = self.cam0_info.K[2] 
        calib_data["value0"]["intrinsics"][0]["intrinsics"]["cy"] = self.cam0_info.K[5] 

        calib_data["value0"]["intrinsics"][1]["intrinsics"]["fx"] = self.cam0_info.K[0] 
        calib_data["value0"]["intrinsics"][1]["intrinsics"]["fy"] = self.cam0_info.K[4] 
        calib_data["value0"]["intrinsics"][1]["intrinsics"]["cx"] = self.cam0_info.K[2] 
        calib_data["value0"]["intrinsics"][1]["intrinsics"]["cy"] = self.cam0_info.K[5] 

        calib_data["value0"]["resolution"] = [[self.cam0_info.height, self.cam0_info.width],
                                              [self.cam1_info.height, self.cam1_info.width]]

        with open(self.basalt_calib_path, 'w') as json_file:
            json.dump(calib_data, json_file, indent = 4)
        print(f"[+] New calib file for basalt code created at {self.basalt_calib_path}.")

    def cam0_data_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='8UC1')
        ts = msg.header.stamp

        f_name = str(self.cam0_path / f"{ts}.png")
        row = [f"{ts}", os.path.basename(f_name)]

        self.cam0_csv_writer.writerow(row)
        cv2.imwrite(f_name, img)

    def cam1_data_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='8UC1')
        ts = msg.header.stamp
        
        f_name = str(self.cam1_path / f"{ts}.png")
        row = [f"{ts}", os.path.basename(f_name)]

        self.cam1_csv_writer.writerow(row)
        cv2.imwrite(f_name, img)

    def rgb_data_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        ts = msg.header.stamp
        f_name = str(self.rgb_path / f"{ts}.png")
        cv2.imwrite(f_name, img)

    def cam0_info_callback(self, msg):
        self.cam0_info = msg
        print(f"Recieved cam0 info.")

    def cam1_info_callback(self, msg):
        self.cam1_info = msg
        print(f"Recieved cam1 info.")

    def cam0_tf_callback(self, msg):
        self.cam0_tf = msg
        print(f"Recieved cam0 tf.")

    def cam1_tf_callback(self, msg):
        self.cam1_tf = msg
        print(f"Recieved cam1 tf.")

    def accel_data_callback(self, msg):
        ts = msg.header.stamp
        self.accel_data[ts] = {
            'a_RS_S_x [m s^-2]': msg.linear_acceleration.x,
            'a_RS_S_y [m s^-2]': msg.linear_acceleration.y,
            'a_RS_S_z [m s^-2]': msg.linear_acceleration.z
        }

    def accel_tf_callback(self, msg):
        self.accel_tf = msg
        print(f"Recieved accel tf.")
    
    def gyro_data_callback(self, msg):
        ts = msg.header.stamp
        self.gyro_data[ts] = {
            'w_RS_S_x [rad s^-1]': msg.angular_velocity.x,
            'w_RS_S_y [rad s^-1]': msg.angular_velocity.y,
            'w_RS_S_z [rad s^-1]': msg.angular_velocity.z
        }

    def gyro_tf_callback(self, msg):
        self.gyro_tf = msg
        print(f"Recieved gyro tf.")

if __name__ == '__main__':
    rospy.init_node('bag2euroc')

    # Initialize parser
    desc = "Python code for conversion of realsense bag file to euroc format."
    parser = argparse.ArgumentParser(description = desc)

    # add arguments
    parser.add_argument("euroc_path", help="path to euroc dataset", type=str)
        
    args = parser.parse_args()
    main = Bag2Euroc(args.euroc_path)
    rospy.spin() 