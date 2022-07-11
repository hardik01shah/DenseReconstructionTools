import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
from mpl_toolkits.mplot3d import Axes3D  
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import FancyArrowPatch
import argparse

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

def generate_poses():
    rad = 3
    f1 = open("trajectories.txt","w")
    for i in range(0, 360, 20):
        r = R.from_euler('zyx', [180+i, 0, 0], degrees=True)
        x = rad*math.cos(math.radians(i))
        y = rad*math.sin(math.radians(i))
        z = 0.0
        f1.write(f"{i} {x} {y} {z} {r.as_quat()[0]} {r.as_quat()[1]} {r.as_quat()[2]} {r.as_quat()[3]}\n")
    f1.close()

def read_poses(pose_path):
    data = np.genfromtxt(pose_path)
    times = data[:,0]
    ts = data[:, 1:4]
    qs = data[:, 4:]
    rs = np.repeat(np.expand_dims(np.eye(4),0), qs.shape[0], 0)

    rs[:, :3, :3] = R.from_quat(qs).as_matrix()
    rs[:, :3, 3] = ts
    poses = rs.copy()
    poses[:, :3, 3] = ts
    poses[:, 3, 3] = 1

    f1 = open("poses.txt","w")
    f1.write(f"{poses}")
    f1.close()

    orig_vec = np.zeros((4,1))
    orig_vec[3,0] = 1

    x_vec = np.zeros((4,1))
    x_vec[0,0] = 1
    x_vec[3,0] = 1
    
    y_vec = np.zeros((4,1))
    y_vec[1,0] = 1
    y_vec[3,0] = 1
    
    z_vec = np.zeros((4,1))
    z_vec[2,0] = 1
    z_vec[3,0] = 1

    plot_data = {}
    points_x = []
    points_y = []
    points_z = []

    for time, pose in zip(times, poses):

        tns = str(time)
        plot_data[tns] = {
            "ox" : 0.0,
            "oy" : 0.0,
            "oz" : 0.0,
            "x_ex" : 0.0,
            "y_ex" : 0.0,
            "z_ex" : 0.0,
            "x_ey" : 0.0,
            "y_ey" : 0.0,
            "z_ey" : 0.0,
            "x_ez" : 0.0,
            "y_ez" : 0.0,
            "z_ez" : 0.0,
        }

        fp = pose @ orig_vec
        plot_data[tns]["ox"] = (fp[0,0])
        plot_data[tns]["oy"] = (fp[1,0])
        plot_data[tns]["oz"] = (fp[2,0])
        points_x.append(fp[0,0])
        points_y.append(fp[1,0])
        points_z.append(fp[2,0])

        xp = pose @ x_vec
        plot_data[tns]["x_ex"] = (xp[0,0]) - (fp[0,0])
        plot_data[tns]["y_ex"] = (xp[1,0]) - (fp[1,0])
        plot_data[tns]["z_ex"] = (xp[2,0]) - (fp[2,0])

        yp = pose @ y_vec
        plot_data[tns]["x_ey"] = (yp[0,0]) - (fp[0,0])
        plot_data[tns]["y_ey"] = (yp[1,0]) - (fp[1,0])
        plot_data[tns]["z_ey"] = (yp[2,0]) - (fp[2,0])

        zp = pose @ z_vec
        plot_data[tns]["x_ez"] = (zp[0,0]) - (fp[0,0])
        plot_data[tns]["y_ez"] = (zp[1,0]) - (fp[1,0])
        plot_data[tns]["z_ez"] = (zp[2,0]) - (fp[2,0])
    
    return plot_data, points_x, points_y, points_z

if __name__=="__main__":

    # to test the visualization script: (generate_poses generates poses in a plane circle around the origin and stores in the trajectories.txt file)
    # generate_poses()
    # pose_path = "trajectories.txt"

    # Initialize parser
    desc = "Python code for visualization of poses obtained from basalt."
    parser = argparse.ArgumentParser(description = desc)

    # add arguments
    parser.add_argument("trajectory_path", help="path to .txt file containing poses from basalt", type=str)
    args = parser.parse_args()
    pose_path = args.trajectory_path

    plot_data, points_x, points_y, points_z = read_poses(pose_path)
    
    setattr(Axes3D, 'arrow3D', _arrow3D)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Trajectory Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-500,500)
    ax.set_ylim(-500,500)
    ax.set_zlim(-500,500)
    
    ax.scatter(points_x, points_y, points_z)
    # ax.plot(points_x, points_y, points_z)
    i=-1
    for time, arrow in plot_data.items():
        i+=1
        if (i%100)!=0:
            continue
        ax.arrow3D(
            arrow["ox"],arrow["oy"],arrow["oz"],
            arrow["x_ex"],arrow["y_ex"],arrow["z_ex"],
            mutation_scale=20,
            arrowstyle="-|>",
            linestyle='dashed',
            fc='red'
            )
        ax.arrow3D(
            arrow["ox"],arrow["oy"],arrow["oz"],
            arrow["x_ey"],arrow["y_ey"],arrow["z_ey"],
            mutation_scale=20,
            arrowstyle="-|>",
            linestyle='dashed',
            fc='green'
            )
        ax.arrow3D(
            arrow["ox"],arrow["oy"],arrow["oz"],
            arrow["x_ez"],arrow["y_ez"],arrow["z_ez"],
            mutation_scale=20,
            arrowstyle="-|>",
            linestyle='dashed',
            fc='blue'
            )
        
    # fig.tight_layout()
    plt.show()