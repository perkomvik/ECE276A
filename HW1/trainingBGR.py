import numpy as np
import matplotlib.pyplot as plt
import os, sys, random, cv2, math

# Gaussian discriminant

random.seed()
dir = os.getcwd()
dir = "/Users/perkvinneslandomvik/PycharmProjects/ECE276A"
trainset = dir + "/trainset/"
labeledset = dir + "/labeledset/mask"
parameters = dir + "/parametersBGR/4/"
bin_masks = dir + "/binmasks4/"

starting_sample = random.randint(0, 45)
trainset_size = 46
num_classes = 2
num_training_samples = 0
all_samples = [x for x in range(trainset_size)]
training_samples = [0 for x in range(num_training_samples)]
testing_samples = [0 for x in range(trainset_size-num_training_samples)]

for i in range(num_training_samples):
    training_samples[i] = all_samples[(starting_sample + i) % 46]
training_samples.sort()
testing_samples = [item for item in all_samples if item not in training_samples]

dimension = (800, 1200)

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
        mask = np.loadtxt(maskpath)
        for row in range(dimension[0]):
            for column in range(dimension[1]):
                x = np.array(img[row, column])
                if mask[row, column] == 1:
                    count += 1
                    numerator_blue = np.add(numerator_blue, x)
                else:
                    numerator_not_blue = np.add(numerator_not_blue, x)
    mu_blue = numerator_blue/count
    mu_not_blue = numerator_not_blue/(len(samples)*dimension[0]*dimension[1]-count)
    return mu_blue, mu_not_blue

def sigma_mle(samples, mu_blue, mu_not_blue):
    count = 0
    numerator_blue = np.zeros((3, 3))
    numerator_not_blue = np.zeros((3, 3))
    for s in samples:
        imagepath = trainset + str(s+1) + ".png"
        maskpath = labeledset + str(s+1) + ".txt"
        img = cv2.imread(imagepath)
        mask = np.loadtxt(maskpath)
        for row in range(dimension[0]):
            for column in range(dimension[1]):
                x = np.array(img[row, column])
                if mask[row, column] == 1:
                    count += 1
                    numerator_blue = np.add(numerator_blue, np.outer((x-mu_blue),(x-mu_blue)))
                else:
                    numerator_not_blue = np.add(numerator_not_blue, np.outer((x-mu_not_blue),(x-mu_not_blue)))
    sigma_blue = numerator_blue/count
    sigma_not_blue = numerator_not_blue/(len(samples)*dimension[0]*dimension[1]-count)
    return sigma_blue, sigma_not_blue

def training():

    theta = theta_mle(training_samples)
    [mu_blue, mu_not_blue] = mu_mle(training_samples)
    [sigma_blue, sigma_not_blue] = sigma_mle(training_samples, mu_blue, mu_not_blue)
    np.savetxt(parameters + "theta.txt", np.array([theta]))
    np.savetxt(parameters + "mu_blue.txt", mu_blue)
    np.savetxt(parameters + "mu_not_blue.txt", mu_not_blue)
    np.savetxt(parameters + "sigma_blue.txt", sigma_blue)
    np.savetxt(parameters + "sigma_not_blue.txt", sigma_not_blue)
    np.savetxt(parameters + "training_samples.txt", np.array(training_samples))

def mahalanobis(x, mu, sigma):

    w_t = x-mu
    res = np.inner(w_t, sigma)
    res2 = np.inner(res, w_t)
    return res2

def test(test_samples):

    # Load parameters
    theta_blue = np.loadtxt(parameters + "theta.txt")
    theta_not_blue = 1-theta_blue
    mu_blue = np.loadtxt(parameters + "mu_blue.txt")
    mu_not_blue = np.loadtxt(parameters + "mu_not_blue.txt")
    sigma_blue = np.loadtxt(parameters + "sigma_blue.txt")
    sigma_not_blue = np.loadtxt(parameters + "sigma_not_blue.txt")
    inv_sigma_blue = np.linalg.inv(sigma_blue)
    inv_sigma_not_blue = np.linalg.inv(sigma_not_blue)
    det_sigma_blue = np.linalg.det(sigma_blue)
    det_sigma_not_blue = np.linalg.det(sigma_not_blue)

    false_positives = 0
    false_negatives = 0
    tot_num_pixels = (len(test_samples) * 800 * 1200)

    for s in test_samples:
        bin_mask = np.zeros((800, 1200))
        imagepath = trainset + str(s+1) + ".png"
        maskpath = labeledset + str(s+1) + ".txt"
        img = cv2.imread(imagepath)
        test_mask = np.loadtxt(maskpath)
        for row in range(dimension[0]):
            for column in range(dimension[1]):
                x = np.array(img[row, column])
                dist_blue = mahalanobis(x, mu_blue, inv_sigma_blue)
                dist_not_blue = mahalanobis(x, mu_not_blue, inv_sigma_not_blue)

                if (2*math.log(theta_blue) - dist_blue - math.log(det_sigma_blue)) > (2*math.log(theta_not_blue) - dist_not_blue - math.log(det_sigma_not_blue)):
                    bin_mask[row, column] = 1
                    if test_mask[row, column] == 0:
                        false_positives += 1

                else:
                    if test_mask[row, column] == 1:
                        false_negatives += 1

        cv2.imwrite(bin_masks + str(s+1) + ".png", bin_mask * 255)
    error_rate = (false_positives+false_negatives)/tot_num_pixels
    false_positives_rate = false_positives/tot_num_pixels
    false_negatives_rate = false_negatives/tot_num_pixels
    return false_positives_rate, false_negatives_rate, error_rate

#training()
[false_positives_rate, false_negative_rate, error_rate] = test(testing_samples)
np.savetxt(parameters + "error_rates2.txt", np.array([false_positives_rate, false_negative_rate, error_rate]))
print("fp  : " + str(false_positives_rate))
print("fn  : " + str(false_negative_rate))
print("tot : " + str(error_rate))




