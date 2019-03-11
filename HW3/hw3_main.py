import numpy as np
from numpy.linalg import multi_dot
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
        self.trajectory[:, :, 0] = np.eye(4)

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
        self.trajectory = trajectory
        self.features = features
        self.n_features = self.features.shape[1]
        self.K = K
        self.b = b
        self.D = np.row_stack((np.eye(3), np.array([0, 0, 0])))
        self.cam_T_imu = cam_T_imu  # In reality this is real frame_T_imu
        self.imu_T_cam = np.linalg.inv(self.cam_T_imu)
        self.M = np.c_[np.vstack((self.K[:2], self.K[:2])), np.array([0, 0, -self.K[0, 0]*self.b, 0])]
        self.landmarks = np.zeros((4, self.n_features))     # TODO: Probably don't initialize all mu
        self.num_landmarks = 0
        self.jacobian = []
        self.sigma = np.empty((0, 0))
        self.V = np.eye(4)*10

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

    def imu_points_to_world(self, w_T_imu, landmarks):
        func = lambda x: np.dot(np.linalg.inv(w_T_imu), x)
        points = np.array([func(elem) for elem in landmarks[:, 1:5]])
        points = np.column_stack((landmarks[:, 0], points))
        return points

    def world_to_camera(self, T, point):
        temp = np.dot(T, point)
        temp2 = np.dot(self.cam_T_imu, temp)
        return temp2

    def dpi_dq(self, q):
        arr = np.array([[1, 0, -q[0]/q[2], 0], [0, 1, -q[1]/q[2], 0], [0, 0, 0, 0], [0, 0, -q[3]/q[2],1]])
        return arr/q[2]

    def update_jacobian(self, T, observations):
        temp = self.world_to_camera(T, self.D)  #o_T_iT_tD in the slides
        H = np.zeros((observations.shape[0]*4, self.num_landmarks*3))
        self.jacobian = np.zeros((observations.shape[0], self.num_landmarks, 4, 3))
        for i,j in enumerate(observations):
            row = 4*i
            col = int(3*j)
            temp2 = self.world_to_camera(T, self.landmarks[:, i])   # o_T_iT_tmu_t,j in the slides
            temp3 = np.dot(self.dpi_dq(temp2), temp)
            self.jacobian[i, int(j)] = np.dot(self.M, temp3)
            H[row:row+4, col:col+3] = np.dot(self.M, temp3)
        # self.jacobian = H
        return H

    def update_kalman_gain(self, H, observations):
        IxV = np.kron(np.eye(observations.shape[0]), self.V)
        temp = multi_dot([H, self.sigma, H.T])
        temp2 = np.linalg.inv(temp + IxV)
        kalman_gain = multi_dot([self.sigma, H.T, temp2])
        return kalman_gain

    def update_mean(self, kalman_gain, active_features, T, t):
        mu_t = self.landmarks[:, :self.num_landmarks]
        func = lambda x: self.world_to_camera(T, x)
        temp = np.apply_along_axis(func, 0, mu_t)
        func2 = lambda y: np.dot(self.M, 1/y[2]*y)
        z_hat = np.apply_along_axis(func2, 0, temp)
        z = self.features[:, active_features, t]
        D = np.kron(np.eye(self.num_landmarks), self.D)
        print(kalman_gain.shape)
        print(D.shape)
        mu_tplus1 = multi_dot([D, kalman_gain, (z-z_hat).flatten()])
        return True

    def kalman_update(self):
        s = 0
        for i in range(3): #t.shape[1]):
            active_features = np.where(features[:, :, i][0, :] > 0)[0]
            imu_points = self.features_to_imu_points(self.features[:, :, i])
            world_points = self.imu_points_to_world(self.trajectory[:, :, i], imu_points)
            if active_features.shape[0]:   # No update if there are no new observations
                new_points = copy.deepcopy(world_points)
                for elem in world_points:   # Checks if this is the first observation of a landmark.
                    if not np.array_equal(self.landmarks[:, elem[0].astype(int)], np.array([0, 0, 0, 0])):
                        index = np.argwhere(new_points[:, 0] == elem[0].astype(int))
                        new_points = np.delete(new_points, index[0, 0], axis=0)
                self.landmarks[:, new_points[:, 0].astype(int)] = new_points[:, 1:].T
                self.num_landmarks = np.count_nonzero(self.landmarks[0, :])
                for _ in range(new_points.shape[0]):
                    s += 1
                    self.sigma = scipy.linalg.block_diag(self.sigma, np.eye(3))

                H = self.update_jacobian(self.trajectory[:, :, i], world_points[:, 0])
                kalman_gain = self.update_kalman_gain(H, world_points[:, 0])
                self.update_mean(kalman_gain, active_features, self.trajectory[:, :, i], i)

        return self.landmarks



if __name__ == '__main__':
    filename = "./data/0042.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)
    # Features are [pixel coords left(x, y), pixel coords rightx(x,y) (starting in top left)][landmark number][timestep]

# (a) IMU Localization via EKF Prediction
    predictor = Predictor(t, linear_velocity, rotational_velocity)
    predictor.test()
    world_T_imu = predictor.trajectory
    traj = predictor.plot

# (b) Landmark Mapping via EKF Update
    updater = Updater(t, world_T_imu, features, K, b, cam_T_imu)
    landmarks = updater.kalman_update()
    points = np.zeros((4, 4, features.shape[1]))
    for i in range(features.shape[1]):
        points[:, :, i] = point_to_transform_3d(landmarks[:, i])
    np.set_printoptions(precision=2)

    # (c) Visual-Inertial SLAM (Extra Credit)

# You can use the function below to visualize the robot pose over time

    visualize_trajectory_2d(traj, points,  show_ori=True) # Feature 76 from dataset 27 is weird
