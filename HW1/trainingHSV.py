import numpy as np
import matplotlib.pyplot as plt
import os, sys, random, cv2, math

# Gaussian discriminant

random.seed()
dir = os.getcwd()
trainset = dir + "/trainset/"
labeledset = dir + "/labeledset1/mask"
parameters = dir + "/parametersHSV/"

starting_sample = 29#random.randint(0, 45)
trainset_size = 46
num_classes = 2
num_training_samples = 35
all_samples = [x for x in range(trainset_size)]
training_samples = [0 for x in range(num_training_samples)]
testing_samples = [0 for x in range(trainset_size-num_training_samples)]

for i in range(num_training_samples):
    training_samples[i] = all_samples[(starting_sample + i) % 46]
training_samples.sort()
testing_samples = [item for item in all_samples if item not in training_samples]

dimension = (800, 1200)

def hue_angle_to_distance(angle1, angle2):
    return -math.fabs(math.fabs(angle1-angle2)-90)+90

def x_minus_mu(x, mu):
    res = np.array([0, 0, 0])
    res[0] = hue_angle_to_distance(x[0], mu[0])
    res[1] = x[1]-mu[1]
    res[2] = x[2]-mu[2]
    return res

def theta_mle(samples):
    bluecount = 0
    for s in samples:
        maskpath = labeledset + str(s+1) + ".txt"
        mask = np.loadtxt(maskpath)
        bluecount += np.count_nonzero(mask)
    pixelcount = len(samples)*dimension[0]*dimension[1]
    return bluecount/pixelcount

def mu_mle(samples):
    count = 0
    numerator_blue = np.zeros((1, 3))
    numerator_not_blue = np.zeros((1, 3))
    for s in samples:
        imagepath = trainset + str(s+1) + ".png"
        maskpath = labeledset + str(s+1) + ".txt"
        img = cv2.imread(imagepath)
        cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = np.loadtxt(maskpath)
        for row in range(dimension[0]):
            for column in range(dimension[1]):
                x = np.array(img[row, column])
                if mask[row, column] == 1:
                    count += 1
                    numerator_blue = np.add(numerator_blue, x)
                numerator_not_blue = np.add(numerator_not_blue, x)
    mu_blue = numerator_blue/count
    mu_not_blue = numerator_not_blue/(len(samples)*dimension[0]*dimension[1]-count)
    return mu_blue, mu_not_blue

def sigma_mle(samples, mu_blue, mu_not_blue):
    count = 0
    numerator_blue = np.zeros((3, 3))
    numerator_not_blue = np.zeros((3, 3))
    for s in samples:
        print(s)
        imagepath = trainset + str(s+1) + ".png"
        maskpath = labeledset + str(s+1) + ".txt"
        img = cv2.imread(imagepath)
        cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = np.loadtxt(maskpath)
        for row in range(dimension[0]):
            for column in range(dimension[1]):
                x = np.array(img[row, column])
                if mask[row, column] == 1:
                    count += 1
                    distance_vector_blue = x_minus_mu(x, mu_blue)
                    numerator_blue = np.add(numerator_blue, np.outer(distance_vector_blue,distance_vector_blue))
                else:
                    distance_vector_not_blue = x_minus_mu(x, mu_not_blue)
                    numerator_not_blue = np.add(numerator_not_blue, np.outer(distance_vector_not_blue, distance_vector_not_blue))
    sigma_blue = numerator_blue/count
    sigma_not_blue = numerator_not_blue/(len(samples)*dimension[0]*dimension[1]-count)
    return sigma_blue, sigma_not_blue

def diag_sigma_mle(samples, mu_blue, mu_not_blue):
    count = 0
    numerator_blue = np.zeros((3, 3))
    numerator_not_blue = np.zeros((3, 3))
    for s in samples:
        print(s)
        imagepath = trainset + str(s + 1) + ".png"
        maskpath = labeledset + str(s + 1) + ".txt"
        img = cv2.imread(imagepath)
        mask = np.loadtxt(maskpath)
        for row in range(dimension[0]):
            for column in range(dimension[1]):
                x = np.array(img[row, column])
                if mask[row, column] == 1:
                    count += 1
                    distance_vector_blue = x_minus_mu(x, mu_blue)
                    numerator_blue = np.add(numerator_blue, np.diag(np.square(distance_vector_blue)))
                else:
                    distance_vector_not_blue = x_minus_mu(x, mu_not_blue)
                    numerator_not_blue = np.add(numerator_not_blue, np.diag(np.square(distance_vector_not_blue)))
    sigma_blue = numerator_blue / count
    sigma_not_blue = numerator_not_blue / (len(samples) * dimension[0] * dimension[1] - count)
    return sigma_blue, sigma_not_blue

def training():
    #theta = theta_mle(training_samples)
    #[mu_blue, mu_not_blue] = mu_mle(training_samples)
    mu_blue = np.loadtxt(parameters + "mu_blue1.txt")
    mu_not_blue = np.loadtxt(parameters + "mu_not_blue1.txt")
    [sigma_blue, sigma_not_blue] = sigma_mle(training_samples, mu_blue, mu_not_blue)
    #np.savetxt(parameters + "theta1.txt", np.array([theta]))
    #np.savetxt(parameters + "mu_blue1.txt", mu_blue)
    #np.savetxt(parameters + "mu_not_blue1.txt", mu_not_blue)
    np.savetxt(parameters + "sigma_blue1.txt", sigma_blue)
    np.savetxt(parameters + "sigma_not_blue1.txt", sigma_not_blue)
    #np.savetxt(parameters + "training_samples1.txt", np.array(training_samples))

def mahalanobis(x, mu, sigma):
    w_t = x_minus_mu(x, mu)
    res = np.inner(w_t, sigma)
    res2 = np.inner(res, w_t)
    return res2

def test(test_samples):
    # Load parameters
    theta_blue = np.loadtxt(parameters + "theta1.txt")
    theta_not_blue = 1-theta_blue
    theta_blue = math.log(theta_blue)
    theta_not_blue = math.log(theta_not_blue)
    mu_blue = np.loadtxt(parameters + "mu_blue1.txt")
    mu_not_blue = np.loadtxt(parameters + "mu_not_blue1.txt")
    sigma_blue = np.loadtxt(parameters + "sigma_blue1.txt")
    sigma_not_blue = np.loadtxt(parameters + "sigma_not_blue1.txt")
    bin_mask = np.zeros((800, 1200))
    errors = 0
    error_rate = 0
    inv_sigma_blue = np.linalg.inv(sigma_blue)
    inv_sigma_not_blue = np.linalg.inv(sigma_not_blue)
    det_sigma_blue = np.linalg.det(sigma_blue)
    det_sigma_not_blue = np.linalg.det(sigma_not_blue)
    det_sigma_blue = math.log(det_sigma_blue)
    det_sigma_not_blue = math.log(det_sigma_not_blue)
    s = 21
    bin_mask = np.zeros((800, 1200))
    imagepath = trainset + str(s+1) + ".png"
    maskpath = labeledset + str(s+1) + ".txt"
    img = cv2.imread(imagepath)
    cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    test_mask = np.loadtxt(maskpath)
    for row in range(dimension[0]):
        for column in range(dimension[1]):
            x = np.array(img[row, column])
            dist_blue = mahalanobis(x, mu_blue, inv_sigma_blue)
            dist_not_blue = mahalanobis(x, mu_not_blue, inv_sigma_not_blue)
            if 2*theta_blue-dist_blue-det_sigma_blue > 2*theta_not_blue-dist_not_blue-det_sigma_not_blue:
                bin_mask[row, column] = 1
            if test_mask[row, column] != bin_mask[row, column]:
                errors += 1
    error_rate = error_rate + errors/(len(test_samples)*800*1200)
    return bin_mask, error_rate

#training()
plt.show(block=False)
[bin_mask, error_rate] = test(testing_samples)
np.savetxt("bin_mask_test.txt", bin_mask)
plt.imshow(bin_mask)
plt.show()
print(error_rate)




