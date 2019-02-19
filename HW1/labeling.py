import os
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from roipoly import RoiPoly

dir = os.getcwd()
trainset = dir + "/trainset/"
labeledset = dir + "/labeledset1/"

trainset_size = 46

# Two classes (blue/not blue)
for i in range(trainset_size):
    # Set image
    img = plt.imread(trainset+str(i+1)+".png")

    # Show the image
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title("left click: line segment         right click: close region")
    plt.show(block=False)

    # Let user draw first ROI
    roi1 = RoiPoly(color='r', fig=fig)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = roi1.get_mask(img)
    maskfile = labeledset + "mask" + str(i+1) + ".txt"
    np.savetxt(maskfile, mask.astype(int), fmt='%d')






