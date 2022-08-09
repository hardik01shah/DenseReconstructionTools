# TUM-VI Rectification

The tum-vi dataset has been recorded from a handheld setup consisting of various sensors i.e. cameras(2), imu and motion-capture.  
The images recorded from the two cameras are distorted as the camera-model is a double-sphere or fisheye model. However, MonoRec works only on rectified images. Hence the tumvi dataset must be rectified. The following opencv functions handle image rectification and have been used for the same:  
1. [seteroRectify()](https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html#gac1af58774006689056b0f2ef1db55ecc)
2. [initUndistortRectifyMap()](https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html#ga0d37b45f780b32f63ed19c21aa9fd333)

In general, the camera intrinsic matrix is a 3x3 of the format:
```
fx 0  cx
0  fy cy
0  0  1
```  

The projection matrix P generated as an output of the [seteroRectify()](https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html#gac1af58774006689056b0f2ef1db55ecc) function is a 3x4 matrix of the format:
```
fx 0  cx fx.tx
0  fy cy 0
0  0  1  0
```  

The camera extrinsics are represented in the homogenous matrix format of 4x4 as:
```
r11 r12 r13 tx 
r21 r22 r23 ty 
r31 r32 r33 tz
0   0   0   1 
```  
The 3x3 top-left matrix is the rotation matrix and the 3x1 last column is the translation vector (tx, ty, tz).  

For running the scripts for tum-vi rectification, create a new conda environment from the given [environment.yml](environment.yml) file using:
```
conda env create -f environment.yml
```

## Scripts
--- 

1. [rectify.py](rectify.py) is used for rectification of a single tum-vi sequence. The script also has the option of saving the updated basalt calibration file using the new intrinsics and extrinsics.
    ```
    usage: rectify.py [-h] [-fov FIELD_OF_VIEW_SCALE] [-sc NEW_SCALE] [-rp RECTIFIED_DIRNAME] [-ip INTRINSIC_PATH] [-bp BASALT_PATH] [-bcn BASALT_CALIB_NAME] dataset_path

    Python code for rectification of TUM-VI sequences. Updated camera intrinsics and extrensics are written to a json file for basalt.

    positional arguments:
    dataset_path          path to tum-vi sequence

    optional arguments:
    -h, --help            show this help message and exit
    -fov FIELD_OF_VIEW_SCALE, --field-of-view-scale FIELD_OF_VIEW_SCALE
                            field of view scale for the new rectified image
    -sc NEW_SCALE, --new-scale NEW_SCALE
                            Scale for the new image size. New image size will be scale times the orig image size.
    -rp RECTIFIED_DIRNAME, --rectified-dirname RECTIFIED_DIRNAME
                            Directory name of the rectified dataset
    -ip INTRINSIC_PATH, --intrinsic-path INTRINSIC_PATH
                            Path for saving the intrinsics file (.yaml). Used only when intrinsics need to be saved without rectifying the entire dataset.
    -bp BASALT_PATH, --basalt-path BASALT_PATH
                            Path to basalt directory. Use to update rectified camera parameters in basalt.
    -bcn BASALT_CALIB_NAME, --basalt-calib-name BASALT_CALIB_NAME
                            Name of the calib file for basalt.

    E.g.
    python3 rectify.py -fov 0.25 -sc 0.5 -rp test ../../tumvi_data/dataset-outdoors1_1024_16
    ```  

2. [rectify_full_tumvi.py](rectify_full_tumvi.py) is used for rectification of the tum-vi dataset i.e. contains several sequences.
    ```
    usage: rectify_full_tumvi.py [-h] [-fov FIELD_OF_VIEW_SCALE] [-sc NEW_SCALE] [-bp BASALT_PATH] dataset_path

    Python code for rectification of TUM-VI sequences. Updated camera intrinsics and extrensics are written to a json file for basalt.

    positional arguments:
    dataset_path          path to tum-vi sequence

    optional arguments:
    -h, --help            show this help message and exit
    -fov FIELD_OF_VIEW_SCALE, --field-of-view-scale FIELD_OF_VIEW_SCALE
                            field of view scale for the new rectified image
    -sc NEW_SCALE, --new-scale NEW_SCALE
                            Scale for the new image size. New image size will be scale times the orig image size.
    -bp BASALT_PATH, --basalt-path BASALT_PATH
                            Path to basalt directory. Use to update rectified camera parameters in basalt.

    E.g.
    python rectify_full_tumvi.py -fov 0.25 -sc 0.5 -bp ../basalt ../../tumvi-dataset/
    ```  

3. [prep_basalt.py](prep_basalt.py) is used for generating a json calibration file for running basalt from a euroc dataset i.e. dso/camchain.yaml
    ```
    usage: prep_basalt.py [-h] tumvi_path basalt_path calib_name

    Python code for using TUM-VI dataset to update camera intrinsics and extrensics for basalt.

    positional arguments:
    tumvi_path   path to tum-vi sequence
    basalt_path  path to basalt directory
    calib_name   name of new calib file for basalt

    optional arguments:
    -h, --help   show this help message and exit

    E.g.
    python3 prep_basalt.py ../../tumvi_data/dataset-outdoors1_512_16_rectified ../basalt test
    ```