import numpy as np
import math
import timeit
from utils import *



class Localizer:
    def __init__(self, t, linear_velocity, rotational_velocity):
        self.t = t
        self.lin_vel = linear_velocity
        self.ang_vel = rotational_velocity
        self.initial_pose = np.zeros(4)
        self.mu = self.initial_pose     # Mean of state prediction
        self.sigma = np.eye(6)          # Variance of state prediction
        self.W = 1*np.eye(6)            # Variance of control input
        self.transform = np.eye(4)#np.linalg.inv(pose_to_transform(self.mu))
        self.prev_t = t[0][0]
        self.trajectory = np.zeros((4, 4, t.shape[1]))

    def prediction(self, timestep):
        current_t = self.t[0][timestep]
        tau = current_t-self.prev_t
        self.prev_t = current_t
        u = np.append(self.lin_vel[:, timestep], self.ang_vel[:, timestep])
        u_hat = hat(u)
        exponent = matrix_exp(tau*u_hat)
        new_transform = np.matmul(exponent, self.transform)
        self.transform = new_transform
        # self.mu = transform_to_pose(new_transform)
        self.trajectory[:, :, timestep] = new_transform

    def test(self):
        for i in range(1, t.shape[1]):
            self.prediction(i)
        return True




if __name__ == '__main__':
    filename = "./data/0042.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)
    # Features are [pixel coords left(x, y), pixel coords rightx(x,y) (starting in top left)][landmark number][timestep]

# (a) IMU Localization via EKF Prediction
    localizer = Localizer(t, linear_velocity, rotational_velocity)
    localizer.test()
    world_T_imu = localizer.trajectory

# (b) Landmark Mapping via EKF Update

# (c) Visual-Inertial SLAM (Extra Credit)

# You can use the function below to visualize the robot pose over time

    visualize_trajectory_2d(world_T_imu, show_ori=True)

