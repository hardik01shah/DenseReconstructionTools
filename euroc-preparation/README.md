# EurocPreparation

The final directory structure for monorec to run on a euroc type dataset must be as shown below. 
The images must be from a camera-model of type _pinhole_. The number _00_ epresents the sequence name. The folders _dso_ and _mav0_ follow the euroc convention.  
_mav0_ contains the data (images, imu etc.) and _dso_ contains sensor information (camera intrinsics, camera extrinsics, distortion parameters, imu intrinsics etc.)

```
dataset-dir
    ├── 00
    |    ├── basalt_keyframe_data
    |    │   ├── keypoints
    |    │   ├── keypoints_viz
    |    │   └── poses
    |    ├── dso
    |    │   ├── cam0
    |    │   │   └── images -> /home/rvlab/Repos/tumvi-dataset/rectified/27/mav0/cam0/data
    |    │   └── cam1
    |    │       └── images -> /home/rvlab/Repos/tumvi-dataset/rectified/27/mav0/cam1/data
    |    ├── mav0
    |        ├── cam0
    |        │   └── data
    |        ├── cam1
    |        │   └── data
    |        ├── imu0
    |        └── mocap0
    ├── 01
    ...
```

The [realsense_bag](realsense_bag) folder consists of the scripts used to prepare the dataset from a bag file recorded from the Realsense camera.

The [tumvi](tumvi) folder consists of the scripts used for rectification and preparation of the TUM-VI dataset.

The [run_basalt_euroc.py](run_basalt_euroc.py) script is used for generating the _basalt_keyframe_data_ folder in the above mentioned directory structure.
```
usage: run_basalt_euroc.py [-h] tumvi_path basalt_path

Python code for running basalt on tumvi sequences

positional arguments:
  tumvi_path   path to tum-vi dataset
  basalt_path  path to basalt directory

optional arguments:
  -h, --help   show this help message and exit

E.g.
python3 run_basalt_tumvi.py ../../tumvi-dataset/ ../../VisualInertialOdometry/
```