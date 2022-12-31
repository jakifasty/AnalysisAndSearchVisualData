#open and read first image, and import SIFT 
import numpy as np
import cv2
#import pyplot as plt
from scipy import ndimage


#read here the image
inputImg = cv2.imread(r"data1/obj1_5.JPG")

#define number of keypoints and other parameters for SIFT algorithm
numOfKeypoints = 100 #
nfeatures = 500
nOctaveLayers = 5
contrastThreshold = 0.13
edgeThreshold = 0.25 #a higher number on this lead to less window keypoint
sigma = 3.2 #minimum sigma that is used in the

#define number of keypoints and other parameters for SURF algorithm


#------apply SIFT keypoint detector algorithm-------

# create SIFT feature extractor
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.13, edgeThreshold = 0.25, nOctaveLayers = 5,  sigma = 3.2)
#
#nfeatures = 500, nOctaveLayers = 5, contrastThreshold = 0.13, edgeThreshold = 0.25, sigma = 3.2

# detect features from the read image
keypoints1, _ = sift.detectAndCompute(inputImg, None)

#draw the keypoints detected 
siftImage = cv2.drawKeypoints(inputImg, keypoints1, inputImg, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite("SIFTImageWithKeypoints.jpg", siftImage)

#------apply SURF keypoint detector algorithm-------

hessianThreshold = 7000
nOctaves = 3
nOctaveLayers = 4

#create SURF feature extractor
surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 7000, nOctaves = 3, nOctaveLayers = 4) #define Hessian threshold to 400

# detect features from the read image
keypoints2, _ = surf.detectAndCompute(inputImg, None)

#draw the keypoints detected
surfImage = cv2.drawKeypoints(inputImg, keypoints2, inputImg, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite("SURFimageWithKeypoints.jpg", surfImage)
