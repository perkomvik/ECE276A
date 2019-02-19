import numpy as np
from math import *
from map_utils import *
from scipy import signal
import helpers
import copy
from operator import itemgetter

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
        self.transform_lidar_body = np.array([[1, 0, -0.0029833/2], [0, 1, 0], [0, 0, 1]])
        self.MAP = {
            "res":     0.1,
            "xmin":    -40,
            "ymin":    -40,
            "xmax":     40,
            "ymax":     40
        }
        self.MAP["sizex"] = int(np.ceil((self.MAP["xmax"] - self.MAP["xmin"]) / self.MAP["res"] + 1))
        self.MAP["sizey"] = int(np.ceil((self.MAP["ymax"] - self.MAP["ymin"]) / self.MAP["res"] + 1))
        self.MAP["map"] = np.zeros((self.MAP["sizex"], self.MAP["sizey"]))
        self.current_cell = self.points_to_cells(0, 0)
        self.trust = log(4)
        self.threshold = 15

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
        self.filter = signal.butter(5, 0.01)
        self.imu = imu
        self.yaw = signal.filtfilt(self.filter[0], self.filter[1], imu["imu_angular_velocity"][2])
        self.encoder = encoder
        self.ratio = self.imu["imu_stamps"].shape[0]/self.encoder["encoder_stamps"].shape[0]  # Number of imu samples per encoder sample
        self.tau = 1/40  # 40 Hz
        self.sd = 0.001# Standard deviation of noise
        self.last_encoder_stamp = 0
        self.offset = 70
        self.MAP = {
            "res":     0.1,
            "xmin":    -40,
            "ymin":    -40,
            "xmax":     40,
            "ymax":     40,
        }
        self.MAP["sizex"] = int(np.ceil((self.MAP["xmax"] - self.MAP["xmin"]) / self.MAP["res"] + 1))
        self.MAP["sizey"] = int(np.ceil((self.MAP["ymax"] - self.MAP["ymin"]) / self.MAP["res"] + 1))
        self.MAP["map"] = np.zeros((self.MAP["sizex"], self.MAP["sizey"]))

    def pose_t_plus_one(self, t, particle):  # New predicted pose with noise added
        # TODO: Fix sync between imu and lidar
        #if t[2] == "imu" and t[1] > self.last_encoder_stamp
        tau = self.tau
        x = particle.pose[0]
        y = particle.pose[1]
        theta = particle.pose[2]
        imu_t = floor((t-1)*self.ratio) + self.offset
        av_yaw_rate = sum(self.yaw[imu_t:ceil(imu_t+self.ratio)])/(ceil(imu_t+self.ratio)-imu_t)
        fr, fl, rr, rl = self.encoder["encoder_counts"][:, t]
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
        weight = particle.weight
        x, y = self.points_to_cells(noised_x, noised_y)

        return Particle(pose, weight)

    def points_to_cells(self, x, y):
        x = np.ceil((x - self.MAP['xmin']) / self.MAP['res']).astype(np.int16) - 1
        y = np.ceil((y - self.MAP['ymin']) / self.MAP['res']).astype(np.int16) - 1
        return x, y

    def best_trajectory(self, particle):
        x, y = self.points_to_cells(particle.pose[0], particle.pose[1])
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
        self.plot(self.MAP)
        return True

class PoseUpdate:
    def __init__(self, lidar):
        self.lidar = lidar
        self.ranges = lidar["lidar_ranges"]
        self.angles = np.arange(-135, 135.25, 0.25) * np.pi / 180.0
        self.transform_lidar_body = np.array([[1, 0, -0.0029833/2], [0, 1, 0], [0, 0, 1]])
        self.MAP = None
        self.N_thresh = 1.1

    def cell_to_point(self, x, y): # x=4, y=4 => 0, 0
        x = (x-4) * self.MAP["res"]*0.25
        y = (y-4) * self.MAP["res"]*0.25
        return x, y

    def weighting(self, t, particles, log_odds):
        self.MAP = copy.deepcopy(log_odds)
        self.MAP["map"][self.MAP["map"] <= 0] = 0
        self.MAP["map"][self.MAP["map"] > 3] = 1
        z_lidar = helpers.lidar_ranges_to_points(self.ranges[:, t+1], self.angles)
        z_body = helpers.transformation(z_lidar, self.transform_lidar_body)
        x_im = np.arange(self.MAP['xmin'], self.MAP['xmax'] + self.MAP['res'], self.MAP['res'])
        y_im = np.arange(self.MAP['ymin'], self.MAP['ymax'] + self.MAP['res'], self.MAP['res'])
        x_range = np.arange(-0.4, 0.4 + 0.1, 0.1)
        y_range = np.arange(-0.4, 0.4 + 0.1, 0.1)
        d_theta = pi/144
        correlations = np.zeros([particles.shape[0]])
        # TODO: Fix this mess
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
                # TODO: Currently only checking corr at current cell (no pose update)
                weight = corr[y_cell][x_cell]
                if theta == 0:
                    main_corr = corr[4][4]
                    correlations[idx] = corr[4][4]
                if weight > max_corr_theta and weight > 1.5*main_corr:  # New point has to have x times the corr as current
                    #print(weight, main_corr)
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
            part.weight = part.weight*correlations[idx]
            new_weights[idx] = part.weight
            if part.weight > largest_weight:
                largest_weight = part.weight
                current_best_particle = part

        self.resample(particles, current_best_particle, new_weights)

        return particles, current_best_particle

    def resample(self, particles, best_particle, new_weights):  # Normalizes and possibly resamples
        norm = sum(new_weights)
        s = 0
        for p in particles:
            p.weight = p.weight/norm
            s += pow(p.weight, 2)
        N_eff = 1/s
        if N_eff < self.N_thresh:
            best_particle.weight = 1/particles.shape[0]
            particles[:] = best_particle







class Particle:

    def __init__(self, pose=np.array([0, 0, 0]), weight=1.0):
        self.pose = pose
        self.weight = weight
        self.transform = self.generate_transform()


    def generate_transform(self):
        t_1 = [cos(self.pose[2]), -sin(self.pose[2]), self.pose[0]]
        t_2 = [sin(self.pose[2]), cos(self.pose[2]), self.pose[1]]
        t_3 = [0, 0, 1]
        return np.array([t_1, t_2, t_3])

if __name__ == "__main__":
    # TODO: Time is not sequential, but epoch time => FIX
    seq = ([i for i in range(lidar["lidar_stamps"].shape[0])] +
           [i for i in range(encoder["encoder_stamps"].shape[0])]+
           [i for i in range(imu["imu_stamps"].shape[0])])
    source = (["lidar" for i in range(lidar["lidar_stamps"].shape[0])] +
              ["encoder" for i in range(encoder["encoder_stamps"].shape[0])] +
              ["imu" for i in range(imu["imu_stamps"].shape[0])])
    stamps = np.concatenate([lidar["lidar_stamps"], encoder["encoder_stamps"], imu["imu_stamps"]])
    readings = list(zip(seq, stamps, source))
    readings = sorted(readings, key=itemgetter(0))

    N = 5
    mapper = Mapper(lidar)
    current_map = mapper.MAP
    predictor = PosePredictor(imu, encoder)
    updater = PoseUpdate(lidar)
    particles = np.array([Particle(weight=1/N) for i in range(N)])
    best_particle = particles[0]

    counter = 0
    for t in readings:#encoder["encoder_counts"].shape[1]-30):
        if t[2] == "lidar":
            current_map = mapper.mapping(t[0], best_particle)
        elif t[2] == "encoder":
            for idx, p in enumerate(particles):
                particles[idx] = predictor.pose_t_plus_one(t[0], particles[idx])  # Might use best_particle
            particles, best_particle = updater.weighting(t[0], particles, current_map)
        #best_particle = particles[0]
        predictor.best_trajectory(best_particle)
        mapper.decay()
        counter += 1
        print(counter)
        if counter > 10000:
            break
    helpers.plot(mapper.MAP, "log_odds.png")
    helpers.plot(predictor.MAP, "trajectory.png")




