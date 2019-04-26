#importing necessay packages for pedestrian detection 

from __future__ import print_function
#from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
import numpy as np
#import argparse

import cv2

## construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--images", required=True, default="D:\kskML\image processing\Pedestrian detection\Pedestrian detection\images")
#args = vars(ap.parse_args())
 
# initialize the HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
for imagePath in paths.list_images("D:\kskML\image processing\Pedestrian detection\Pedestrian detection\images"):
    print(imagePath)
    image = cv2.imread(imagePath)
    img = image.copy()
    #image = cv2.resize(image,(500,780))
    orig = imutils.resize(image, width=min(400, image.shape[1]))
   

    # detect people in the image
    rects, weights = hog.detectMultiScale(orig, winStride=(6, 6), padding = (8, 8), scale=1.05)
    
    if(len(rects) == 0):
        print("no human")

    else:
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("After NMS", orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()