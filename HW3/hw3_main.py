import numpy as np
from utils import *


class Localizer:
    def __init__(self, t, linear_velocity, rotational_velocity):
        self.t = t
        self.v = linear_velocity
        self.w = rotational_velocity

    def test(self):
        print(self.t.shape)
        print(self.v.shape)
        print(self.w.shape)



if __name__ == '__main__':
    filename = "./data/0042.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)

# (a) IMU Localization via EKF Prediction

    l = Localizer(t, linear_velocity, rotational_velocity)


# (b) Landmark Mapping via EKF Update

# (c) Visual-Inertial SLAM (Extra Credit)

# You can use the function below to visualize the robot pose over time

# visualize_trajectory_2d(world_T_imu,show_ori=True)

