import numpy as np
from math import *
from map_utils import *
from scipy import signal
import helpers
import copy
import cv2
from PIL import Image
from matplotlib import cm
from operator import itemgetter
from datetime import datetime

exec(open("load_data.py").read())

lidar = {
    "lidar_angle_min": lidar_angle_min,
    "lidar_angle_max": lidar_angle_max,
    "lidar_angle_increment": lidar_angle_increment,
    "lidar_range_min": lidar_range_min,
    "lidar_range_max": lidar_range_max,
    "lidar_ranges": lidar_ranges,
    "lidar_stamps": lidar_stamps,
    "lidar_ranges_per_scan": int((lidar_angle_max - lidar_angle_min) / lidar_angle_increment[0][0]) + 1
}

imu = {
    "imu_angular_velocity": imu_angular_velocity,
    "imu_stamps": imu_stamps
}

encoder = {
    "encoder_counts": encoder_counts,
    "encoder_stamps": encoder_stamps
}

kinect = {
    "disp_stamps": disp_stamps,
    "rgb_stamps": rgb_stamps
}


class Mapper:
    def __init__(self, lidar):
        self.lidar = lidar
        self.ranges = lidar["lidar_ranges"]
        self.angles = np.arange(-135, 135.25, 0.25)*np.pi/180.0
        self.transform_lidar_body = np.array([[1, 0, -0.13323], [0, 1, 0], [0, 0, 1]])
        self.MAP = {
            "res":     0.1,
            "xmin":    -35,
            "ymin":    -35,
            "xmax":     35,
            "ymax":     35
        }
        self.MAP["sizex"] = int(np.ceil((self.MAP["xmax"] - self.MAP["xmin"]) / self.MAP["res"] + 1))
        self.MAP["sizey"] = int(np.ceil((self.MAP["ymax"] - self.MAP["ymin"]) / self.MAP["res"] + 1))
        self.MAP["map"] = np.zeros((self.MAP["sizex"], self.MAP["sizey"]))
        self.current_cell = self.points_to_cells(0, 0)
        self.trust = log(2)
        self.threshold = 10

    def points_to_cells(self, x, y):
        x = np.ceil((x - self.MAP['xmin']) / self.MAP['res']).astype(np.int16) - 1
        y = np.ceil((y - self.MAP['ymin']) / self.MAP['res']).astype(np.int16) - 1
        return x, y

    def mapping(self, t, particle):
        self.current_cell = self.points_to_cells(particle.pose[0], particle.pose[1])
        particle.transform = particle.generate_transform()
        ranges = self.ranges[:, t]
        lidar_points = helpers.lidar_ranges_to_points(ranges, self.angles)
        body_points = helpers.transformation(lidar_points, self.transform_lidar_body)
        world_points = helpers.transformation(body_points, particle.transform)
        occupied_x, occupied_y = self.points_to_cells(world_points[0], world_points[1])
        for i in range(occupied_x.shape[0]):
            clear_x, clear_y = np.array(bresenham2D(self.current_cell[0], self.current_cell[1], occupied_x[i], occupied_y[i]), dtype=np.int16)
            self.MAP["map"][clear_x[:-1], clear_y[:-1]] -= self.trust
        self.MAP["map"][occupied_x, occupied_y] += self.trust
        np.clip(self.MAP["map"], -self.threshold, self.threshold, out=self.MAP["map"])
        return self.MAP

    def decay(self):
        self.MAP["map"]*0.9

    def test(self):
        self.mapping(0, Particle())
        helpers.plot(self.MAP, "mapper_test.png")



class PosePredictor:
    def __init__(self, imu, encoder):
        self.filter = signal.butter(5, 0.1)
        self.imu = imu
        self.yaw = signal.filtfilt(self.filter[0], self.filter[1], imu["imu_angular_velocity"][2])
        self.encoder = encoder
        self.ratio = self.imu["imu_stamps"].shape[0]/self.encoder["encoder_stamps"].shape[0]  # Number of imu samples per encoder sample
        self.tau = 1/40  # 40 Hz
        self.sd = 0.001  # Standard deviation of noise
        self.last_encoder_stamp = 0
        self.last_imu_readings = []
        self.MAP = {
            "res":     0.1,
            "xmin":    -35,
            "ymin":    -35,
            "xmax":     35,
            "ymax":     35,
        }
        self.MAP["sizex"] = int(np.ceil((self.MAP["xmax"] - self.MAP["xmin"]) / self.MAP["res"] + 1))
        self.MAP["sizey"] = int(np.ceil((self.MAP["ymax"] - self.MAP["ymin"]) / self.MAP["res"] + 1))
        self.MAP["map"] = np.zeros((self.MAP["sizex"], self.MAP["sizey"]))
        self.trajectory = []

    def pose_t_plus_one(self, t, particle):  # New predicted pose with noise added
        # TODO: Fix sync between imu and lidar
        if t[2] == "imu":
            if t[1] > self.last_encoder_stamp:
                self.last_imu_readings.append(self.yaw[t[0]])
                return particle
            else:
                print("somethings fucky")
        elif t[2] == "encoder":
            tau = self.tau
            x = particle.pose[0]
            y = particle.pose[1]
            theta = particle.pose[2]
            if len(self.last_imu_readings) == 0:
                av_yaw_rate = 0
            else:
                av_yaw_rate = sum(self.last_imu_readings)/float(len(self.last_imu_readings))
            fr, fl, rr, rl = self.encoder["encoder_counts"][:, t[0]]
            dist_l = (fl+rl)/2 * 0.0022
            dist_r = (fr+rr)/2 * 0.0022
            v_l = dist_l/tau
            v_r = dist_r/tau
            v = (v_l+v_r)/2
            delta_x = v * tau * np.sinc(av_yaw_rate*tau/2) * cos(theta+av_yaw_rate*tau/2)
            delta_y = v * tau * np.sinc(av_yaw_rate*tau/2) * sin(theta+av_yaw_rate*tau/2)
            delta_theta = tau * av_yaw_rate
            x = x + delta_x
            y = y + delta_y
            theta = theta + delta_theta
            noised_x = x + np.random.normal(0, self.sd, 1)[0]
            noised_y = y + np.random.normal(0, self.sd, 1)[0]
            noised_theta = theta + np.random.normal(0, self.sd/2, 1)[0]
            pose = np.array([noised_x, noised_y, noised_theta])
            particle.pose = pose
            particle.transform = particle.generate_transform()
            self.last_imu_readings = []
            return particle
        else:
            print("lidar scan entered")
            return False



    def points_to_cells(self, x, y):
        x = np.ceil((x - self.MAP['xmin']) / self.MAP['res']).astype(np.int16) - 1
        y = np.ceil((y - self.MAP['ymin']) / self.MAP['res']).astype(np.int16) - 1
        return x, y

    def best_trajectory(self, particle, t):
        x, y = self.points_to_cells(particle.pose[0], particle.pose[1])
        self.trajectory.append((particle.pose, t))
        self.MAP["map"][x, y] = 1

    def test(self, particle):
        new_particle = particle
        x = np.zeros(encoder["encoder_counts"].shape[1])
        y = np.zeros(encoder["encoder_counts"].shape[1])
        for i in range(encoder["encoder_counts"].shape[1]):
            new_particle = self.pose_t_plus_one(i, new_particle)
            x[i] = new_particle.pose[0]
            y[i] = new_particle.pose[1]
        x, y = self.points_to_cells(x, y)
        self.MAP["map"][x, y] = 1
        helpers.plot(self.MAP)
        return True

class PoseUpdate:
    def __init__(self, lidar):
        self.lidar = lidar
        self.ranges = lidar["lidar_ranges"]
        self.angles = np.arange(-135, 135.25, 0.25) * np.pi / 180.0
        self.transform_lidar_body = np.array([[1, 0, -0.0029833/2], [0, 1, 0], [0, 0, 1]])
        self.N_thresh = 1.1

        self.MAP = {
            "res":     0.1,
            "xmin":    -35,
            "ymin":    -35,
            "xmax":     35,
            "ymax":     35,
        }
        self.MAP["sizex"] = int(np.ceil((self.MAP["xmax"] - self.MAP["xmin"]) / self.MAP["res"] + 1))
        self.MAP["sizey"] = int(np.ceil((self.MAP["ymax"] - self.MAP["ymin"]) / self.MAP["res"] + 1))
        self.MAP["map"] = np.zeros((self.MAP["sizex"], self.MAP["sizey"]))

    def points_to_cells(self, x, y):
        x = np.ceil((x - self.MAP['xmin']) / self.MAP['res']).astype(np.int16) - 1
        y = np.ceil((y - self.MAP['ymin']) / self.MAP['res']).astype(np.int16) - 1
        return x, y

    def cell_to_point(self, x, y): # x=4, y=4 => 0, 0
        x = (x-4) * self.MAP["res"]*0.25
        y = (y-4) * self.MAP["res"]*0.25
        return x, y

    def weighting(self, t, particles, log_odds):
        self.MAP = copy.deepcopy(log_odds)
        self.MAP["map"][self.MAP["map"] <= 0] = 0
        self.MAP["map"][self.MAP["map"] > 0] = 1
        z_lidar = helpers.lidar_ranges_to_points(self.ranges[:, t], self.angles)
        z_body = helpers.transformation(z_lidar, self.transform_lidar_body)
        x_im = np.arange(self.MAP['xmin'], self.MAP['xmax'] + self.MAP['res'], self.MAP['res'])
        y_im = np.arange(self.MAP['ymin'], self.MAP['ymax'] + self.MAP['res'], self.MAP['res'])
        x_range = np.arange(-0.4, 0.4 + 0.1, 0.1)
        y_range = np.arange(-0.4, 0.4 + 0.1, 0.1)
        d_theta = pi/72
        correlations = np.zeros([particles.shape[0]])
        for idx, part in enumerate(particles):
            best_x = 0
            best_y = 0
            best_theta = 0
            max_corr_theta = -1000
            main_corr = -1000
            part_copy = copy.deepcopy(part)
            for i in [0]:  # add variance to lidar angle: 0° ,-2.5°, -1.25°, 1.25°, 2.5°
                theta = i * d_theta
                part_copy.pose[2] = part.pose[2]+theta
                part_copy.transform = part_copy.generate_transform()
                z_world = helpers.transformation(z_body, part_copy.transform)
                z = np.stack((z_world[0], z_world[1]))
                corr = mapCorrelation(self.MAP["map"], x_im, y_im, z, x_range, y_range)
                x_cell, y_cell = np.unravel_index(np.argmax(corr, axis=None), corr.shape)

                x, y = self.cell_to_point(x_cell, y_cell)
                weight = corr[y_cell][x_cell]
                if theta == 0:
                    main_corr = corr[4][4]
                    correlations[idx] = corr[4][4]
                if weight > max_corr_theta and weight > 10000*main_corr:  # New point has to have x times the corr as current
                    max_corr_theta = weight
                    correlations[idx] = weight
                    best_x = x
                    best_y = y
                    best_theta = theta
            part.pose[0] = part.pose[0] + best_x
            part.pose[1] = part.pose[1] + best_y
            part.pose[2] = part.pose[2] + best_theta

        correlations = helpers.softmax(correlations)
        new_weights = np.zeros([particles.shape[0]])
        largest_weight = 0
        current_best_particle = Particle()
        for idx, part in enumerate(particles):
            part.weight = correlations[idx]
            new_weights[idx] = part.weight
            if part.weight > largest_weight:
                largest_weight = part.weight
                current_best_particle = part

        self.resample(particles, current_best_particle, new_weights)

        return particles, current_best_particle

    def resample(self, particles, best_particle, new_weights):  # Normalizes and possibly resamples
        norm = sum(new_weights)
        s = 0
        for i, p in enumerate(particles):
            p.weight = new_weights[i]/norm
            s += pow(p.weight, 2)
        N_eff = 1/s
        if N_eff < self.N_thresh:
            best_particle.weight = 1/particles.shape[0]
            particles[:] = best_particle

class TextureMapper:
    def __init__(self, kinect):

        self.kinect = kinect
        self.rotation = np.linalg.inv(helpers.rotation_matrix(roll=0, pitch=0.36, yaw=0.021))
        self.translation = np.array([0.18, 0.005, 0.36+0.254/2])
        self.translation.shape = (3, 1)
        self.T_bc = np.vstack((np.hstack((self.rotation, self.translation)), [0, 0, 0, 1]))
        self.intrinsics = np.linalg.inv(np.array([[585.05108211, 0, 242.94140713], [0, 585.05108211, 315.83800193], [0, 0, 1]]))
        self.canonical = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        self.R_co = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        self.MAP = 0

    def disparity_to_depth(self, d):
        dd = -0.00304 * d + 3.31
        depth = 1.03/dd
        return depth

    def ir_to_rgb(self, i, j, d):
        depth = self.disparity_to_depth(d)
        rgbi = (i * 526.37 + 1.03/depth * -4.5 * 1750.46 + 19276.0)/585.051
        rgbj = (j * 526.37 + 16662)/585.051
        return rgbi, rgbj

    def pixel_to_body(self, u, v, d):
        depth = self.disparity_to_depth(d)
        if depth < 0.05:
            return np.array([0, 0, 0, 0])
        pixel = np.array([u, v, 1])
        optical_frame = depth * np.dot(self.intrinsics, pixel)
        regular_frame = np.dot(self.R_co, optical_frame)
        body_frame = np.dot(self.T_bc, np.append(regular_frame, 1))
        return body_frame


    def test(self, p):
        p1 = self.pixel_to_body(10, 10, 2)
        p2 = self.pixel_to_body(300, 400, 1)
        transform = p.transform_3d()
        print(p1)
        world = helpers.transformation_3d(p1, transform)
        print(world)







class Particle:

    def __init__(self, pose=np.array([0, 0, 0]), weight=1.0):
        self.pose = pose
        self.weight = weight
        self.transform = self.generate_transform()


    def generate_transform(self):
        t1 = [cos(self.pose[2]), -sin(self.pose[2]), self.pose[0]]
        t2 = [sin(self.pose[2]), cos(self.pose[2]), self.pose[1]]
        t3 = [0, 0, 1]
        return np.array([t1, t2, t3])

    def transform_3d(self):
        t1 = [self.transform[0][0], self.transform[0][1], 0, self.transform[0][2]]
        t2 = [self.transform[1][0], self.transform[1][1], 0, self.transform[1][2]]
        t3 = np.append(self.transform[2], 0)
        t4 = [0, 0, 0, 1]
        return np.array([t1, t2, t3, t4])

if __name__ == "__main__":
    # seq = ([i for i in range(lidar["lidar_stamps"].shape[0])] +
    #        [i for i in range(encoder["encoder_stamps"].shape[0])]+
    #        [i for i in range(imu["imu_stamps"].shape[0])])
    # source = (["lidar" for i in range(lidar["lidar_stamps"].shape[0])] +
    #           ["encoder" for i in range(encoder["encoder_stamps"].shape[0])] +
    #           ["imu" for i in range(imu["imu_stamps"].shape[0])])
    # stamps = np.concatenate([lidar["lidar_stamps"], encoder["encoder_stamps"], imu["imu_stamps"]])
    # readings = list(zip(seq, stamps, source))
    # readings = sorted(readings, key=itemgetter(1))
    #
    mapper = Mapper(lidar)
    texture_mapper = TextureMapper(kinect)
    predictor = PosePredictor(imu, encoder)
    updater = PoseUpdate(lidar)

    task = "texture_map"

    if task == "slam":
        N = 1
        particles = np.array([Particle(weight=1 / N) for i in range(N)])
        best_particle = particles[0]
        current_map = mapper.MAP
        counter = 0
        for reading in readings:
            if reading[2] == "lidar":
                print(counter)
                # if counter > 1000:
                #     break
                counter += 1
                current_map = mapper.MAP
                particles, best_particle = updater.weighting(reading[0], particles, current_map)
                updated_map = mapper.mapping(reading[0], best_particle)
            else:
                for idx, p in enumerate(particles): # Particles remain unchanged if t is an IMU reading, waits for encoder to update
                    particles[idx] = predictor.pose_t_plus_one(reading, particles[idx])
            if reading[2] == "encoder":
                predictor.best_trajectory(best_particle, reading[1])
            mapper.decay()
        np.save("pose_trajectory.npy", np.array(predictor.trajectory))
        np.save("occupancy_grid.npy", np.array(updater.MAP["map"]))
        helpers.plot(mapper.MAP, "log_odds2.png")

    elif task == "texture_map":
        predictor.trajectory = np.load("pose_trajectory.npy")
        occupancy_grid = np.load("occupancy_grid.npy")
        n_channels = 3
        size = updater.MAP["map"].shape[0]
        im = Image.fromarray(np.uint8(cm.gist_earth(occupancy_grid)*255))
        im.save("texture_map.png")
        texture_map = cv2.imread("texture_map.png")
        counter = 0
        particle = Particle()
        timestamps = [x[1] for x in predictor.trajectory]
        for i, t in enumerate(kinect["disp_stamps"]):
            if i % 10 != 0:
                continue
            if counter >= 50:
                break
            counter += 1
            print(i)
            idx = (np.abs(timestamps - t)).argmin()
            particle.pose = predictor.trajectory[idx][0]
            particle.transform = particle.generate_transform()
            disp_img = cv2.imread(
                "dataRGBD/Disparity" + str(dataset) + "/disparity" + str(dataset) + "_" + str(i+1) + ".png", -1)
            rgb_img = cv2.imread("dataRGBD/RGB"+str(dataset)+"/rgb"+str(dataset)+"_"+str(i+1)+".png")
            print("img: " + str(i+1))
            for v, row in enumerate(disp_img[350:]):  # Starts at upper left
                for u, disp in enumerate(row):
                    coord = texture_mapper.pixel_to_body(u, v+350, disp)
                    if coord.any():
                        if coord[2] < 0.3:
                            rgbi, rgbj = np.floor(texture_mapper.ir_to_rgb(u, v+350, disp)).astype(int)  # i,j starts in bottom left
                            color = rgb_img[rgbj][rgbi]
                            x, y = helpers.transformation(([coord[0]], [coord[1]]), particle.transform)
                            x, y = predictor.points_to_cells(x[0], y[0])
                            texture_map[x][y] = np.flip(color)
        print(2)
        plt.imsave("test.png", texture_map)
        plt.imshow(texture_map, cmap="hot")
        plt.show()



        # updater.MAP["map"] = np.load("occupancy_grid.npy")
        # for item in predictor.trajectory:
        #     x, y = predictor.points_to_cells(item[0][0], item[0][1])
        #     predictor.MAP["map"][x, y] = 1
        # helpers.plot(predictor.MAP, "trajectory2.png")
        # helpers.plot(updater.MAP, "occupancy.png")






