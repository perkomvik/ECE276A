# HW3 - Visual Inertial SLAM

Project implementing the Extended Kalman Filter (EKF) for estimating the trajectory of an IMU located in a car and positions of landmarks captured by a stereo camera setup.

### Files:
* `hw3_main.py`: Implements part a) and b) of task 4 in the assignment. Contains the classes `Predictor` and `Updater` that solves problems a) and b) respectively.
* `utils.py`: Contains helper functions for transformations and hat-mapping among others. Also contains an altered version of `visualize_trajectory_2d` that also plots the landmark positions, and not only the trajectory of the IMU.

### Classes: 
* `class Predictor`: Implements the prediction step for the EKF in estimating the pose of the IMU. 
* `class Updates`: Implements the update step for the EKF in estimating the 3D positions (x,y,z) of the features given as stereo pixel coordinates. Referred to as landmarks in the code.

More specific function descriptions are found in the files themselves.