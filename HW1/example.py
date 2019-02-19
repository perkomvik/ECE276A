import logging

import numpy as np
import cv2
import matplotlib as mpl
mpl.use("TKAgg")
from matplotlib import pyplot as plt

from roipoly import RoiPoly

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                    level=logging.INFO)

# Create image
img = plt.imread("trainset/1.png")

# Show the image
fig = plt.figure()
plt.imshow(img)
plt.title("left click: line segment         right click: close region")
plt.show(block=False)

# Let user draw first ROI
roi1 = RoiPoly(color='r', fig=fig)

# Show the image with the first ROI
fig = plt.figure()
plt.imshow(img)
plt.colorbar()
roi1.display_roi()
plt.title('draw second ROI')
plt.show(block=False)

# Let user draw second ROI
roi2 = RoiPoly(color='b', fig=fig)

# Show the image with both ROIs and their mean values
plt.imshow(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

[x.display_roi() for x in [roi1, roi2]]
[x.display_mean(img) for x in [roi1, roi2]]
plt.title('The two ROIs')
plt.show()


# Show ROI masks
plt.imshow(roi1.get_mask(img) + roi2.get_mask(img),
           interpolation='nearest', cmap="Greys")
plt.title('ROI masks of the two ROIs')
plt.show()