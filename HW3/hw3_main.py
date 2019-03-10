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
        self.pose = self.initial_pose   # Mean of state prediction
        self.sigma = np.eye(6)          # Variance of state prediction
        self.W = 1*np.eye(6)            # Variance of control input
        self.mu = np.linalg.inv(pose_to_transform(self.pose))
        self.prev_t = t[0][0]
        self.trajectory = np.zeros((4, 4, t.shape[1]))
        self.plot = np.zeros((4, 4, t.shape[1]))
        self.plot[:, :, 0] = np.eye(4)

    def predict_pose(self, u, tau):  # Input: new IMU reading, output: prediction of next pose
        u_hat = hat(u)
        exponent = matrix_exp(-tau*u_hat)
        return np.matmul(exponent, self.mu)

    def predict_sigma(self, u, tau):
        u_hat = hat(u)
        u_adj = adj(u_hat)
        exponent = matrix_exp(-tau*u_adj)
        temp = np.dot(self.sigma, exponent.T)
        sigma = np.dot(exponent, temp) + tau**2*self.W
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
        self.n_features = self.features.shape[1]
        self.K = K
        self.b = b
        self.cam_T_imu = cam_T_imu  # In reality this is real frame_T_imu
        self.imu_T_cam = np.linalg.inv(self.cam_T_imu)
        self.M = np.c_[np.vstack((self.K[:2], self.K[:2])), np.array([0, 0, -self.K[0, 0]*self.b, 0])]
        self.landmarks = np.zeros((self.n_features, 4)) # TODO: Probably don't initialize all mu

    def features_to_imu_points(self, features: "features at current time"):   # Converts stereo-camera pixels [ur, vr, ul, vl] to IMU frame coordinates
        active_features = np.where(features[0, :] > 0)[0]
        fsub = -self.M[2, 3]
        fsu = self.M[0, 0]
        fsv = self.M[1, 1]
        cu = self.M[0, 2]
        cv = self.M[1, 2]
        new_landmarks = np.zeros((active_features.shape[0], 5))
        for i, f in enumerate(active_features):
            z = fsub/(features[0, f]-features[2, f])
            x = (features[0, f] - cu)*z/fsu
            y = (features[1, f] - cv)*z/fsv
            point_imu = np.dot(self.imu_T_cam, np.array([x, y, z, 1]))
            elem = np.append(np.array(f), point_imu)
            new_landmarks[i] = elem
        return new_landmarks

    def test(self):
        imu_points = self.features_to_imu_points(self.features[:, :, 0])
        self.landmarks[imu_points[:, 0].astype(int)] = imu_points[:, 1:]
        print(self.landmarks)
        # TODO: Convert imu to world using trajectory



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
