from numpy.linalg import multi_dot
import scipy.linalg
import copy
from utils import *


class Predictor:
    """Performs the Kalman prediction step for the IMU pose using velocity readings from IMU"""
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
        self.plot[:, :, 0] = np.eye(4)          # Tracks the pose of the IMU in the world frame for plotting
        self.trajectory[:, :, 0] = np.eye(4)    # Tracks the transform from world to IMU to simplify the update step

    def predict_pose(self, u, tau):
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

    def kalman_predict(self):
        """Main loop for the prediction step"""
        for timestep in range(1, t.shape[1]):
            current_t = self.t[0][timestep]
            tau = current_t-self.prev_t
            self.prev_t = current_t

            u = np.append(self.lin_vel[:, timestep], self.ang_vel[:, timestep])
            self.mu = self.predict_pose(u, tau)
            self.pose = transform_to_pose(self.mu)
            self.sigma = self.predict_sigma(u, tau)

            self.trajectory[:, :, timestep] = self.mu
            self.plot[:, :, timestep] = np.linalg.inv(self.mu)


class Updater:
    """Performes the Kalman update step for estimating the 3D locations of the features input"""
    def __init__(self, t, trajectory, features, K, b, cam_T_imu):
        self.t = t  # Epoch time of each timestep
        self.trajectory = trajectory    # The trajectory estimated in the prediction step. Assumed correct
        self.features = features
        self.n_features = self.features.shape[1]
        self.K = K  # Intrinsic parameters of left camera
        self.b = b  # x-axis offset between left and right camera
        self.D = np.row_stack((np.eye(3), np.array([0, 0, 0])))  # Perturbation matrix
        self.cam_T_imu = cam_T_imu  # In reality this is (real frame)_T_imu because axes are flipped
        self.imu_T_cam = np.linalg.inv(self.cam_T_imu)

        # Complete intrinsic matrix for stereo camera setup
        self.M = np.c_[np.vstack((self.K[:2], self.K[:2])), np.array([0, 0, -self.K[0, 0]*self.b, 0])]
        self.landmarks = np.zeros((4, self.n_features))  # List of current estimates of landmarks
        self.num_landmarks = 0  # Keeps track of the total observed landmarks up to a time t
        self.sigma = np.empty((0, 0))  # Covariance matrix for the landmarks
        self.V = np.eye(4)*1  # Observation noise

    def features_to_imu_points(self, features: "features at current time"):
        """Converts stereo-camera pixels [ur, vr, ul, vl] to IMU frame coordinates"""
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
        """Transforms points from the IMU frame to world frame using the current transform w_T_imu"""
        func = lambda x: np.dot(np.linalg.inv(w_T_imu), x)
        points = np.array([func(elem) for elem in landmarks[:, 1:5]])
        points = np.column_stack((landmarks[:, 0], points))
        return points

    def world_to_camera(self, T, point):
        temp = np.dot(T, point)
        temp2 = np.dot(self.cam_T_imu, temp)
        return temp2

    def dpi_dq(self, q):
        """Derivative of the projection function pi(q) = 1/q*q"""
        arr = np.array([[1, 0, -q[0]/q[2], 0], [0, 1, -q[1]/q[2], 0], [0, 0, 0, 0], [0, 0, -q[3]/q[2],1]])
        return arr/q[2]

    def update_jacobian(self, T, observations):
        """Updates the Jacobian matrix according to slide 7 in lecture 15"""
        temp = self.world_to_camera(T, self.D)  #o_T_iT_tD in the slides
        H = np.zeros((observations.shape[0]*4, self.num_landmarks*3))
        for i, j in enumerate(observations):
            row = 4*i
            col = int(3*j)
            temp2 = self.world_to_camera(T, self.landmarks[:, i])   # o_T_iT_tmu_t,j in the slides
            temp3 = np.dot(self.dpi_dq(temp2), temp)
            H[row:row+4, col:col+3] = np.dot(self.M, temp3)
        return H

    def update_kalman_gain(self, H, observations):
        """Computes the new Kalman gain using the Jacobian matrix"""
        IxV = np.kron(np.eye(observations.shape[0]), self.V)
        temp = multi_dot([H, self.sigma, H.T])
        temp2 = np.linalg.inv(temp + IxV)
        kalman_gain = multi_dot([self.sigma, H.T, temp2])
        return kalman_gain

    def update_mu(self, kalman_gain, active_features, T, t):
        """Computes new estimates for the positions of the landmarks"""
        mu = self.landmarks[:, active_features]
        func = lambda x: self.world_to_camera(T, x)
        temp = np.apply_along_axis(func, 0, mu)
        func2 = lambda y: np.dot(self.M, 1/y[2]*y)
        z_hat = np.apply_along_axis(func2, 0, temp)
        z = self.features[:, active_features, t]
        D = np.kron(np.eye(self.num_landmarks), self.D)
        temp = multi_dot([D, kalman_gain, (z - z_hat).flatten()]).reshape(4, self.num_landmarks)
        new_mu = self.landmarks[:, :self.num_landmarks] + temp
        self.landmarks[:, :self.num_landmarks] = new_mu
        return True

    def update_sigma(self, H, kalman_gain):
        """Computes the new estimate for the covariance of the landmarks"""
        shape = kalman_gain.shape[0]
        I = np.eye(shape)
        temp = I-np.dot(kalman_gain, H)
        new_sigma = np.dot(temp, self.sigma)
        self.sigma = new_sigma
        return True

    def kalman_update(self):
        """Main update loop"""
        for i in range(t.shape[1]):
            active_features = np.where(features[:, :, i][0, :] > 0)[0]  # Keeps track of the observed features at time t
            imu_points = self.features_to_imu_points(self.features[:, :, i])
            world_points = self.imu_points_to_world(self.trajectory[:, :, i], imu_points)

            # No update if there are no new observations
            if active_features.shape[0]:
                new_points = copy.deepcopy(world_points)
                for elem in world_points:
                    # Checks if this is the first observation of a landmark.
                    if not np.array_equal(self.landmarks[:, elem[0].astype(int)], np.array([0, 0, 0, 0])):
                        index = np.argwhere(new_points[:, 0] == elem[0].astype(int))
                        new_points = np.delete(new_points, index[0, 0], axis=0)
                # Initialize landmarks first observed at current time
                self.landmarks[:, new_points[:, 0].astype(int)] = new_points[:, 1:].T
                self.num_landmarks = np.count_nonzero(self.landmarks[0, :])
                # Expand and initialize new dimensions of sigma with new landmarks added
                for _ in range(new_points.shape[0]):
                    self.sigma = scipy.linalg.block_diag(self.sigma, np.eye(3))
                H = self.update_jacobian(self.trajectory[:, :, i], world_points[:, 0])
                kalman_gain = self.update_kalman_gain(H, world_points[:, 0])
                self.update_mu(kalman_gain, active_features, self.trajectory[:, :, i], i)
                self.update_sigma(H, kalman_gain)
        return self.landmarks


if __name__ == '__main__':
    filename = "./data/0020.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)
    # Features are [pixel coords left(x, y), pixel coords rightx(x,y) (starting in top left)][landmark number][timestep]

# (a) IMU Localization via EKF Prediction
    predictor = Predictor(t, linear_velocity, rotational_velocity)
    predictor.kalman_predict()
    world_T_imu = predictor.trajectory
    traj = predictor.plot

# (b) Landmark Mapping via EKF Update
    updater = Updater(t, world_T_imu, features, K, b, cam_T_imu)
    landmarks = updater.kalman_update()

    # Only rejects outliers that are far from the median positions in the x,y and z directions
    filtered_landmarks = remove_outliers(landmarks)
    points = np.zeros((4, 4, filtered_landmarks.shape[1]))
    for i in range(filtered_landmarks.shape[1]):
        points[:, :, i] = point_to_transform_3d(filtered_landmarks[:, i])

    visualize_trajectory_2d(traj, points,  show_ori=True)
