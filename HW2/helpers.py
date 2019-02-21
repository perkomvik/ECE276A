import numpy as np
import matplotlib.pyplot as plt
from math import *
from datetime import datetime

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def transformation(points, transform):
    ones = np.ones(len(points[0]))
    points = np.vstack([points, ones])
    func = lambda x: np.inner(x, transform)
    transformed_points = np.apply_along_axis(func, 0, points)
    return transformed_points[:2]

def transformation_3d(point, transform):
    return np.matmul(transform, point)


def rotation_matrix(roll, pitch, yaw):
    r1 = [cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll) + sin(yaw)*cos(roll), -cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll)]
    r2 = [-sin(yaw)*cos(pitch), -sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll) + cos(yaw)*sin(roll)]
    r3 = [sin(pitch), -cos(pitch)*sin(roll), cos(pitch)*cos(roll)]
    return np.array([r1, r2, r3])

def lidar_ranges_to_points(ranges, angles):
    valid_ranges = np.logical_and((ranges < 30), (ranges > 0.1))
    ranges = ranges[valid_ranges]
    angles = angles[valid_ranges]
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return x, y

def exectime(t):
    dt = datetime.now()
    dt = dt.microsecond
    print(str(dt-t) + " ms")



def plot(MAP, filename):
    plt.interactive(False)
    plt.figure()
    plt.imsave(filename, MAP["map"], cmap="gray")