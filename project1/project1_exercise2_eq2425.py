import numpy as np
import cv2
#import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from matplotlib import pyplot as plt
#import vlfeat

#----Image Feature Matching----
#read here the images
inputImg1 = cv2.imread(r"data1/obj1_5.JPG")
inputImg2 = cv2.imread(r"data1/obj1_t1.JPG")

#queryImg = cv2.imread(r"data1/obj1_t1.JPG")
#databaseImg = cv2.imread(r"data1/obj1_t1.JPG")

section = "b"

if section == "a": # a) Extract few hundred SIFT features from the test images, using vl_feat: vl_sift. Show feature keypoints superimposed on top of the JPG images

    #------apply SIFT keypoint detector algorithm-------

    x1 = 200
    y1 = 100
    x2 = 200
    y2 = 100

    # compute SIFTs with OpenCV
    sift1 = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.13, edgeThreshold = 0.25, nOctaveLayers = 5,  sigma = 3.2)
    sift2 = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.13, edgeThreshold = 0.25, nOctaveLayers = 5,  sigma = 3.2)
    #sift2 = cv2.xfeatures2d.SIFT_create(nfeatures=500, nOctaveLayers=3, contrastThreshold = 0.04, edgeThreshold=10, sigma=1.6)

    # find the keypoints and descriptors with SIFT
    list_kp1, _ = sift1.detectAndCompute(inputImg1,None)
    list_kp2, _ = sift2.detectAndCompute(inputImg2,None)

    #draw the keypoints detected 
    siftQueryImage = cv2.drawKeypoints(inputImg1, list_kp1, inputImg1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    siftDatabaseImage = cv2.drawKeypoints(inputImg2, list_kp2, inputImg2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #cv2.imshow(match_img)
    #cv2.waitKey()

    #plt.subplot(121), plt.imshow(siftQueryImage)
    #plt.title('Query image with keypoints')
    #plt.subplot(122), plt.imshow(siftQueryImage)
    #plt.title('Database image with keypoints')
    #plt.show()

    finalImg = None
    finalImg = np.concatenate((siftQueryImage, siftDatabaseImage), axis=1)
    cv2.imwrite(f"SIFT_Images_With_Keypoints_Ex2a.JPG", finalImg)

    #save also images individually
    cv2.imwrite(f"SIFT_QueryImages_With_Keypoints_Ex2a.JPG", siftQueryImage)
    cv2.imwrite(f"SIFT_DatabaseImage_With_Keypoints_Ex2a.JPG", siftDatabaseImage)

    #save the two resulting images together with the keypoints
    #cv2.imwrite("SIFTQueryImageWithKeypoints.jpg", siftQueryImage)
    #cv2.imwrite("SIFTDatabaseImageWithKeypoints.jpg", siftDatabaseImage)

    #plt.savefig('SIFT_matched_image_exercise2.png', bbox_inches='tight')



if section == "b": #b) Implement the "fixed threshold" matching algorithm. Adjust distance threshold until you obtain a satisfying result

    doubleWithKeypoints = cv2.imread(r"SIFT_Images_With_Keypoints_Ex2a.JPG")

    threshold = 0.2
    # compute SIFTs with OpenCV
    sift1 = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.13, edgeThreshold = 0.25, nOctaveLayers = 5,  sigma = 3.2)
    sift2 = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.13, edgeThreshold = 0.25, nOctaveLayers = 5,  sigma = 3.2)

    # find the keypoints and descriptors with SIFT
    list_kp1, descr1 = sift1.detectAndCompute(inputImg1,None)
    list_kp2, descr2 = sift2.detectAndCompute(inputImg2,None)

    #draw the keypoints detected 
    siftQueryImage = cv2.drawKeypoints(inputImg1, list_kp1, inputImg1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    siftDatabaseImage = cv2.drawKeypoints(inputImg2, list_kp2, inputImg2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #calculate distance to every point
    good_matches = []
    for idx1, vectorDesc1 in enumerate(descr1):
        for idx2, vectorDesc2 in enumerate(descr2):
            d = np.linalg.norm(vectorDesc1 - vectorDesc2)
            if d < 500.0: #check which point has the closest distance
                #add to list of matches
                #good_matches.append([list_kp1[idx1], list_kp2[idx2], d])
                good_matches.append(cv2.DMatch(_imgIdx=0, _queryIdx=idx1, _trainIdx=idx2,_distance=0) for idx1 in range(len(good_matches)))
    
    #draw the matches
    print(good_matches)
    for i in range(len(good_matches)):
        kp1 = good_matches[i][0]
        kp2 = good_matches[i][1]
        cv2.line(doubleWithKeypoints, (int(kp1.pt[0]), int(kp1.pt[1])), (doubleWithKeypoints.shape[1] + int(kp2.pt[0]), int(kp2.pt[1])), (255,0,0), 1)
    
    #save the two resulting images together with the keypoints
    cv2.imwrite("SIFT_Images_With_Matches_Ex2b.jpg", doubleWithKeypoints)
    #matched_img = cv2.drawMatches(inputImg1, list_kp1, inputImg2, list_kp2, good_matches, None, flags=2)

    #match_img = cv2.drawMatches(siftQueryImage, list_kp1, siftDatabaseImage, list_kp2, matches, None, flags=2)

    # Draw first 10 matches.
    #match_img = cv2.drawMatches(siftQueryImage, list_kp1, siftDatabaseImage, list_kp2, matches[:10], None, flags=2)

    #save the full mached image
    #cv2.imwrite(f"SIFT_Images_With_Matches_Ex2b.JPG", matched_img)



if section == 'c': 


    # BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

#if section == 'd':
    
#if section == 'e':

