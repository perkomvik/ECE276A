import numpy as np
import math
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler


def load_data(file_name):
    """
    function to read visual features, IMU measurements and calibration parameters
    Input:
      file_name: the input data file. Should look like "XXX_sync_KLT.npz"
    Output:
      t: time stamp
          with shape 1*t
      features: visual feature point coordinates in stereo images,
          with shape 4*n*t, where n is number of features
      linear_velocity: IMU measurements in IMU frame
          with shape 3*t
      rotational_velocity: IMU measurements in IMU frame
          with shape 3*t
      K: (left)camera intrinsic matrix
          [fx  0 cx
            0 fy cy
            0  0  1]
          with shape 3*3
      b: stereo camera baseline
          with shape 1
      cam_T_imu: extrinsic matrix from IMU to (left)camera, in SE(3).
          close to
          [ 0 -1  0 t1
            0  0 -1 t2
            1  0  0 t3
            0  0  0  1]
          with shape 4*4
    """
    with np.load(file_name) as data:
        t = data["time_stamps"]  # time_stamps
        features = data["features"]  # 4 x num_features : pixel coordinates of features
        linear_velocity = data["linear_velocity"]  # linear velocity measured in the body frame
        rotational_velocity = data["rotational_velocity"]  # rotational velocity measured in the body frame
        K = data["K"]  # intrindic calibration matrix
        b = data["b"]  # baseline
        cam_T_imu = data["cam_T_imu"]  # Transformation from imu to camera frame
    return t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu


def visualize_trajectory_2d(pose, points, path_name="Unknown", show_ori=False):
    """
    Function to visualize the trajectory and landmarks in 2D
    Input:
        pose: 4*4*N matrix representing the camera pose,
              where N is the number of pose, and each
              4*4 matrix is in SE(3)

        points: 4x4*M matrix representing the landmarks
                positions, where M is the number of
                landmarks, and each 4x4 matrix is in SE(3)
                with all rotations set to zero
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    n_pose = pose.shape[2]
    ax.plot(points[0, 3, :], points[1, 3, :], 'g.', linestyle="None", markersize=5, fillstyle="full")
    ax.plot(pose[0, 3, :], pose[1, 3, :], 'r-', label=path_name, linewidth=1)
    ax.scatter(pose[0, 3, 0], pose[1, 3, 0], marker='s', label="start")
    ax.scatter(pose[0, 3, -1], pose[1, 3, -1], marker='o', label="end")
    if show_ori:
        select_ori_index = list(range(0, n_pose, int(n_pose/50)))
        yaw_list = []
        for i in select_ori_index:
            _, _, yaw = mat2euler(pose[:3, :3, i])
            yaw_list.append(yaw)
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx, dy = [dx, dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0, 3, select_ori_index], pose[1, 3, select_ori_index], dx, dy,
                  color="b", units="xy", width=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.savefig('trajectory.eps', format='eps', dpi=1000)
    plt.show(block=False)
    return fig, ax

def pose_to_transform(pose):
    """Pose in 6D space (3D position and orientation)"""
    R = rotation_matrix(pose[3], pose[4], pose[5])
    p = np.array([pose[0], pose[1], pose[2]])
    T = np.c_[R, p]
    bot_row = np.array([0, 0, 0, 1])
    T = np.r_[T, [bot_row]]
    return T

def point_to_transform_3d(point):
    """Point in 3D space to homogeneous transformation matrix. Used for plotting landmarks"""
    R = np.eye(3)
    p = np.array([point[0], point[1], point[2]])
    T = np.c_[R, p]
    bot_row = np.array([0, 0, 0, 1])
    T = np.r_[T, [bot_row]]
    return T

def rotation_matrix(roll, pitch, yaw):
    """Generate rotation matrix SO(3) for given values for roll, pitch and yaw in sequence xyz"""
    r1 = [math.cos(yaw)*math.cos(pitch), math.cos(yaw)*math.sin(pitch)*math.sin(roll) + math.sin(yaw)*math.cos(roll),
          -math.cos(yaw)*math.sin(pitch)*math.cos(roll) + math.sin(yaw)*math.sin(roll)]
    r2 = [-math.sin(yaw)*math.cos(pitch), -math.sin(yaw)*math.sin(pitch)*math.sin(roll) + math.cos(yaw)*math.cos(roll),
          math.sin(yaw)*math.sin(pitch)*math.cos(roll) + math.cos(yaw)*math.sin(roll)]
    r3 = [math.sin(pitch), -math.cos(pitch)*math.sin(roll), math.cos(pitch)*math.cos(roll)]
    return np.array([r1, r2, r3])


def transform_to_pose(transform):
    """Convert the 4x4 transform to 6D pose with position and orientation"""
    roll = np.arctan2(transform[2][1], transform[2][2])
    pitch = np.arctan2(transform[2][0], math.sqrt(transform[2][1]**2+transform[2][2]**2))
    yaw = np.arctan2(transform[1][0], transform[0][0])
    return np.append(transform[:3, 3], np.array([roll, pitch, yaw]))


def skew(v):
    """Generate skew symmetric matrix from vector. Only works on 3x1 vector"""
    a, b, c = v[0], v[1], v[2]
    if v.shape != (3, 1) and v.shape != (3,):
        raise Exception('shape of vector has to be (3, 1). The shape of vector was: {}'.format(v.shape))
    return np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])


def hat(u):
    """Maps a 6x1 vector to a 4x4 matrix in se(3). Used for velocity vector u with [ang_vel, lin_vel]^T"""
    v = u[:3]
    omega = u[3:]
    omega_hat = skew(omega)
    u_hat = np.c_[omega_hat, v]
    u_hat = np.r_[u_hat, [np.zeros(4)]]
    return u_hat

def adj(u_hat):
    """Adjoint of u_hat"""
    omega_hat = u_hat[:3, :3]
    v_hat = skew(u_hat[:3, 3])
    temp = np.hstack((omega_hat, v_hat))
    temp2 = np.hstack((np.zeros((3, 3)), omega_hat))
    u_adj = np.vstack((temp, temp2))
    return u_adj

def matrix_exp(matrix):
    """Computes a simple matrix exponential using only the first two terms of the exponential formula"""
    shape = matrix.shape
    if shape[0] != shape[1]:
        raise Exception('The matrix has to be square, shape was {}'.format(shape))
    res = np.eye(shape[0])+matrix
    return res

def remove_outliers(data, m=6):
    """Removes outliers that are far away from the others. Mainly to make plots prettier"""
    med = np.median(data, axis=1)
    func = lambda x: x-med
    d = np.abs(np.apply_along_axis(func, 0, data))
    mdev = np.median(d, axis=1)
    func2 = lambda x: x/mdev
    s = np.apply_along_axis(func2, 0, d)
    res = np.empty((4, 1))
    for idx, point in enumerate(data.T):
        if s[0, idx] < m and s[1, idx] < m:
            res = np.c_[res, point]
    return res[:, 1:]

