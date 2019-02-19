'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import numpy as np
import math

class BarrelDetector():
    def __init__(self):
        '''
            Initilize your blue barrel detector with the attributes you need
            eg. parameters of your classifier
        '''
        self.theta_blue = 1.096330357142857176e-02
        self.theta_not_blue = 1-self.theta_blue
        self.mu_blue = np.array([1.229473025542461784e+02, 6.604512619208560409e+01, 2.762207798201250242e+01])
        self.mu_not_blue = np.array([8.969141681963085944e+01, 1.008037406106404745e+02, 1.088228030503345991e+02])
        self.sigma_blue = np.array([[2.819416671518373278e+03, 1.443964795949852714e+03, 2.073161243482311704e+02],
                                   [1.443964795949852714e+03, 1.068351890552637997e+03, 4.591018826271445050e+02],
                                   [2.073161243482311704e+02, 4.591018826271445050e+02, 6.767154277231610422e+02]])
        self.sigma_not_blue = np.array([[3.736715415316265080e+03, 3.526852137576529003e+03, 3.401670168098522936e+03],
                                   [3.526852137576529003e+03, 3.539246495361252528e+03, 3.512956902567480029e+03],
                                   [3.401670168098522936e+03, 3.512956902567480029e+03, 3.778228825865411181e+03]])
        self.inv_sigma_blue = np.linalg.inv(self.sigma_blue)
        self.inv_sigma_not_blue = np.linalg.inv(self.sigma_not_blue)
        self.det_sigma_blue = np.linalg.det(self.sigma_blue)
        self.det_sigma_not_blue = np.linalg.det(self.sigma_not_blue)


    def segment_image(self, img):
        '''
            Calculate the segmented image using a classifier
            eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
        '''
    # YOUR CODE HERE
        def mahalanobis(x, mu, sigma):
            w_t = x - mu
            res = np.inner(w_t, sigma)
            res2 = np.inner(res, w_t)
            return res2
        mask_img = np.zeros((800, 1200))

        for row in range(800):
            for column in range(1200):
                x = np.array(img[row, column])
                dist_blue = mahalanobis(x, self.mu_blue, self.inv_sigma_blue)
                dist_not_blue = mahalanobis(x, self.mu_not_blue, self.inv_sigma_not_blue)
                if (2 * math.log(self.theta_blue) - dist_blue - math.log(self.det_sigma_blue)) > \
                        (2 * math.log(self.theta_not_blue) - dist_not_blue - math.log(self.det_sigma_not_blue)):
                    mask_img[row, column] = 1
        return mask_img



    def get_bounding_box(self, img):
        '''
            Find the bounding box of the blue barrel
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                is from left to right in the image.

            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        '''
        # YOUR CODE HERE

        import skimage.morphology
        mask_img = self.segment_image(img)
        A = np.ones((20, 5))

        mask_img = skimage.morphology.binary_closing(mask_img, A)
        mask_img = label(mask_img)
        props = regionprops(mask_img)

        boxes = []
        for r in props:
            if r.solidity > 0.7 and r.area > 500 and r.area < 50000:  # Remove very small or very big areas
                print(r.area)
                bounding_box = r.bbox
                x1 = bounding_box[1]
                x2 = bounding_box[3]
                y1 = bounding_box[0]
                y2 = bounding_box[2]
                if (y2 - y1) > 1.2 * (x2 - x1) and (y2 - y1) < 4*(x2 - x1):  # Make sure that we get an upright box with reasonable dimensions
                    boxes.append([x1, y1, x2, y2])
        largest_area = 1
        largest_prop = 0
        if not boxes:  # If no suitable boxes found, take a chance on the largest "blue" region
            for r in props:
                if r.area > largest_area:
                    largest_area = r.area
                    largest_prop = r
            bounding_box = largest_prop.bbox
            x1 = bounding_box[1]
            x2 = bounding_box[3]
            y1 = bounding_box[0]
            y2 = bounding_box[2]
            boxes.append([x1, y1, x2, y2])
        print(boxes)
        return boxes


if __name__ == '__main__':
    folder = "trainset"
    my_detector = BarrelDetector()
    for filename in os.listdir(folder):
        # read one test image
        img = cv2.imread(os.path.join(folder, filename))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #Display results:
        #(1) Segmented images
        mask_img = my_detector.segment_image(img)
        #(2) Barrel bounding box
        boxes = my_detector.get_bounding_box(img)

    #The autograder checks your answers to the functions segment_image() and get_bounding_box()
    #Make sure your code runs as expected on the testset before submitting to Gradescope

