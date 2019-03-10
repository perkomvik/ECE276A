import numpy as np
import math
import timeit
import copy

from utils import *



class Predictor:
    def __init__(self, t, linear_velocity, rotational_velocity):
        self.t = t
        self.lin_vel = linear_velocity
        self.ang_vel = rotational_velocity
        self.initial_pose = np.zeros(6)
        self.pose = self.initial_pose     # Mean of state prediction
        self.sigma = np.eye(6)          # Variance of state prediction
        self.W = 1*np.eye(6)            # Variance of control input
        self.mu = np.linalg.inv(pose_to_transform(self.pose))
        self.prev_t = t[0][0]
        self.trajectory = np.zeros((4, 4, t.shape[1]))
        self.plot = np.zeros((4, 4, t.shape[1]))
        self.plot[:, :, 0] = np.eye(4)

    def predict_pose(self, u, tau):  #Takes in new IMU reading to predict the next pose
        u_hat = hat(u)
        exponent = matrix_exp(-tau*u_hat)
        return np.matmul(exponent, self.mu)

    def predict_sigma(self, u, tau):
        u_hat = hat(u)
        u_adj = adj(u_hat)
        exponent = matrix_exp(-tau*u_adj)
        temp = np.dot(self.sigma, exponent.T)
        sigma = np.dot(exponent, temp) + tau**2*self.W
        np.set_printoptions(precision=2)
        return sigma



    def prediction(self, timestep):
        current_t = self.t[0][timestep]
        tau = current_t-self.prev_t
        self.prev_t = current_t
        u = np.append(self.lin_vel[:, timestep], self.ang_vel[:, timestep])

        self.mu = self.predict_pose(u, tau)
        self.pose = transform_to_pose(self.mu)
        self.sigma = self.predict_sigma(u, tau)

        self.trajectory[:, :, timestep] = self.mu
        self.plot[:, :, timestep] = np.linalg.inv(self.mu)

    def test(self):
        for i in range(1, t.shape[1]):
            self.prediction(i)
        return True


class Updater:
    def __init__(self, t, trajectory, features, K, b, cam_T_imu):
        self.t = t
        self.trajecotry = trajectory
        self.features = features
        self.K = K
        self.b = b
        self.cam_T_imu = cam_T_imu  # In reality this is real frame_T_imu
        self.M = np.c_[np.vstack((self.K[:2], self.K[:2])),np.array([0, 0, -self.K[0, 0]*self.b, 0])]

    def test(self):
        active_features = np.where(self.features[0, :, 0] > 0)[0]
        fsub = -self.M[2,3]
        fsu = self.M[0, 0]
        fsv = self.M[1, 1]
        cu = self.M[0, 2]
        cv = self.M[1, 2]
        for f in active_features:
            z = fsub/(features[0, f, 0]-features[2, f, 0])
            x = (features[0, f, 0] - cu)*z/fsu
            y = (features[1, f, 0] - cv)*z/fsv
            print(x,y,z)


if __name__ == '__main__':
    filename = "./data/0027.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)
    # Features are [pixel coords left(x, y), pixel coords rightx(x,y) (starting in top left)][landmark number][timestep]

# (a) IMU Localization via EKF Prediction
    predictor = Predictor(t, linear_velocity, rotational_velocity)
    predictor.test()
    world_T_imu = predictor.trajectory

# (b) Landmark Mapping via EKF Update
    updater = Updater(t, world_T_imu, features, K, b, cam_T_imu)
    updater.test()

# (c) Visual-Inertial SLAM (Extra Credit)

# You can use the function below to visualize the robot pose over time

    visualize_trajectory_2d(world_T_imu, show_ori=True)

