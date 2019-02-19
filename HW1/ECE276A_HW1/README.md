## Project 1
Description of submitted files.

### labeling.py
Responsible for labeling. Loops through images and waits for user input for the regions of interest. Limited to one polygon per image.

### training.py
Where the models for the gaussian distributions are trained. Also where testing of a model is done on a selectable subset of the training data, and binary masks are created for the segmented images. 

### bounding_box.py
Loads binary masks and performs the barrel detection.

### barrel_detector.py
Incorporates the base functionality of the "testing" part of training.py.
Initialises parameters already found, then uses these in the segmentation.
get_bounding_box() also performs segmentation and outputs the coordinates for the bounding box. 
