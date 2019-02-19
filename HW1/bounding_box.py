import numpy as np
import cv2
from skimage.measure import label, regionprops
import skimage.morphology
import skimage.draw

def get_bounding_box(img): # Almost identical to barrel_detector.py but draws boxes and writes to png

    A = np.ones((10, 10))
    B = np.zeros((20, 20))
    nb = B.shape[0]
    na = A.shape[0]
    lower = (nb) // 2 - (na // 2)
    upper = (nb // 2) + (na // 2)
    na2 = A.shape[1]
    B[lower:upper, lower:upper] = A

    A = np.ones((20,5))

    img = skimage.morphology.binary_closing(img, A)
    img = label(img)
    cv2.imwrite("morphed.png", img*255)
    props = regionprops(img)
    barrels = []
    boxes = []
    largest_area = 1
    for r in props:
        if r.area > largest_area:
            largest_area = r.area
            largest_prop = r
        if r.solidity > 0.7 and r.area > 550 and r.area < 50000:
            print(r.area)
            bounding_box = r.bbox
            x1 = bounding_box[1]
            x2 = bounding_box[3]
            y1 = bounding_box[0]
            y2 = bounding_box[2]
            if (y2 - y1) > 1.2 * (x2 - x1) and (y2 - y1) < 4*(x2 - x1):
                boxes.append([x1, y1, x2, y2])
                barrels.append(r)
    if not boxes:
        bounding_box = largest_prop.bbox
        x1 = bounding_box[1]
        x2 = bounding_box[3]
        y1 = bounding_box[0]
        y2 = bounding_box[2]
        boxes.append([x1, y1, x2, y2])
        barrels.append(largest_prop)
    final_img = np.zeros((800, 1200))
    for barrel in barrels:
        for i in range(len(barrel.coords)):
            final_img[barrel.coords[i][0]][barrel.coords[i][1]] = 1
            cv2.rectangle(final_img, (barrel.bbox[1], barrel.bbox[0]), (barrel.bbox[3], barrel.bbox[2]), (255, 255, 0))

    cv2.imwrite("output.png", final_img*255)
    print(boxes)

img = cv2.imread("binmasks4/1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
get_bounding_box(img)



