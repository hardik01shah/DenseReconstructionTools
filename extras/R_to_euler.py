import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA

if __name__=="__main__":
    tm_1 = np.array(
        [[ 1.0000,  0.0077, -0.0136,  0.5124],
        [-0.0077,  1.0003,  0.0016, -0.0026],
        [ 0.0136, -0.0016,  0.9999, -0.0791],
        [ 0.0000,  0.0000,  0.0000,  1.0000]]
    )

    r_tm_1 = R.from_matrix(tm_1[:3,:3])
    t_tm_1 = tm_1[:3,3]
    print(f"Frame1")
    print(f"Roll Pitch Yaw (TUM-VI): {r_tm_1.as_euler('zyx', degrees=True)}")
    print(f"Norm of translation (TUM-VI): {LA.norm(t_tm_1)}")

    tm_2 = np.array(
        [[ 9.9669e-01, -7.7266e-02,  3.2850e-02, -4.4069e-01],
        [ 7.7304e-02,  9.9700e-01, -6.2820e-05, -5.5261e-02],
        [-3.2741e-02,  2.5175e-03,  9.9969e-01,  5.4765e-02],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]
    )

    r_tm_2 = R.from_matrix(tm_2[:3,:3])
    t_tm_2 = tm_2[:3,3]
    print(f"Frame2")
    print(f"Roll Pitch Yaw (TUM-VI): {r_tm_2.as_euler('zyx', degrees=True)}")
    print(f"Norm of translation (TUM-VI): {LA.norm(t_tm_2)}")

    kt_1 = np.array(
        [[ 9.9997e-01, -5.2213e-04,  2.7445e-03,  1.0021e-02],
        [ 5.2962e-04,  1.0004e+00, -3.5474e-03, -1.8936e-02],
        [-2.7379e-03,  3.5582e-03,  9.9985e-01,  7.6388e-01],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]
    )

    r_kt_1 = R.from_matrix(kt_1[:3,:3])
    t_kt_1 = kt_1[:3,3]
    print(f"Frame1")
    print(f"Roll Pitch Yaw (KITTI): {r_kt_1.as_euler('zyx', degrees=True)}")
    print(f"Norm of translation (KITTI): {LA.norm(t_kt_1)}")

    kt_2 = np.array(
        [[ 1.0001, -0.0032, -0.0018, -0.0076],
        [ 0.0033,  1.0003,  0.0073,  0.0099],
        [ 0.0017, -0.0073,  0.9999, -0.7699],
        [ 0.0000,  0.0000,  0.0000,  1.0000]]
    )

    r_kt_2 = R.from_matrix(kt_2[:3,:3])
    t_kt_2 = kt_2[:3,3]
    print(f"Frame2")
    print(f"Roll Pitch Yaw (KITTI): {r_kt_2.as_euler('zyx', degrees=True)}")
    print(f"Norm of translation (KITTI): {LA.norm(t_kt_2)}")