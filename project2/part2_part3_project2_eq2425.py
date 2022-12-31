#use SIFT descriptors as image features, use vocabulary trees as database structures
from ctypes import sizeof
from functools import _Descriptor
from sil import Sil
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image
import glob

from tqdm import tgdm


#a) load database images and extract 400 SIFT features from it
for obj_num in range(1, 51): #this iterates over all the images in the database (50 in total)
    comb_kp = [] #this will store the keypoints of all the images
    comb_des = [] #this will store the descriptors of all the images
    for img_num in range(1, 3): #this iterates over all the images of each building (3 for each)
        #read the image
        databaseImg = cv.imread(f"data2/databaseImages/obj{obj_num}_{img_num}.JPG")

        #extract SIFT features from the database images
        sift = cv.xfeatures2d.SIFT_create(contrastThreshold = 0.04, edgeThreshold=10, sigma=3.2)

        #find the keypoints and descriptors of the image with SIFT
        kp, descr = sift.detectAndCompute(databaseImg, None)

        comb_kp.append(kp) #add the keypoints to the array
        comb_des.append(descr) #add the descriptors to the array
    
    #here we combine the features from the same object and we save them in the array
    totalDatabaseFeatures = np.vstack((comb_kp, comb_des))
    totalSum = totalDatabaseFeatures.sum(axis=1)
    print(totalSum)
    averageNumberOfDFeatures = np.mean(totalSum)
    print("Average number of features: ", averageNumberOfDFeatures)


#b) load query images and extract 400 SIFT features from it
for obj_num in range(1, 51): #this iterates over all the images in the database (50 in total)
    comb_kp = [] #this will store the keypoints of all the images
    comb_des = [] #this will store the descriptors of all the images
    for img_num in range(1, 2): #this iterates over all the images of each building (3 for each)
        #read the image
        print(obj_num, img_num)
        queryImg = cv.imread(f"data2/queryImages/obj{obj_num}_{img_num}.JPG")

        #extract SIFT features from the database images
        sift = cv.xfeatures2d.SIFT_create(contrastThreshold = 0.04, edgeThreshold=10, sigma=3.2)

        #find the keypoints and descriptors of the image with SIFT
        kp, descr = sift.detectAndCompute(queryImg, None)

        comb_kp.append(kp) #add the keypoints to the array
        comb_des.append(descr) #add the descriptors to the array
        
    #here we combine the features from the same building into one array
    totalQueryFeatures = np.vstack((comb_kp, comb_des))
    averageNumberOfQFeatures = np.mean(totalQueryFeatures)
    print("Average number of features: ", averageNumberOfQFeatures)



#3. Vocabulary tree construction
data = totalDatabaseFeatures #SIFT features from the database images
b = None #brunch number
depth_tree = None #dept_tree (int)

depth = None
childrens = [] #empty list of children for every node
feature_vector = np.array([]) #empty feature vector for every node
is_leaf = None #boolean value to check if the node is a leaf or not
leaf_idx = None #index number of the leaf
leafs = [] #empty list of leafs
idf_values = None 
n_samples = None #total number of samples
obj_indices = None #number of the object

class Node:
    def __init__(self, depth, childrens, feature_vector, is_leaf, leaf_idx, idf_values, n_samples, obj_indices=None):
        self.depth = depth #depth of the node
        self.childrens = childrens #list of children for every node
        self.feature_vector = feature_vector
        self.is_leaf = is_leaf #boolean value to check if the node is a leaf or not
        self.leaf_idx = leaf_idx #index number of the leaf
        self.idf_values = idf_values #inverse document frequency value for every object
        self.n_samples = n_samples #number of training samples in that node cluster
        self.obj_indices = obj_indices #number of the object of the 

class VocabularyTree:
    def __init__(self):
        self.root = root #root node of the tree
        self.data = data
        self.b = b
        self.depth_tree = depth_tree
        self.idx2leaf = {} #list translating the index of the leaf to the leaf itself

        #statistics
        self.n_leaves = 0 #number of leaves in the tree when it is created
        self.n_nodes_per_level = 0 #number of nodes for each level of the tree when it is created
        self.n_samples = 0 #number of samples in the tree when it is created
        self.n_features = 0 #number of features in the tree when it is created
        self.n_objects = 0 #number of objects in the tree when it is created

    def fit(self, data, b, depth_tree):

        ##this builds a hierarchical tree with the given data, branches and depth
        self.data = data
        self.b = b
        self.depth_tree = depth_tree

        obj_indices = [data[i].shape[0] for i in range(len(data))] #number of the object of the
        obj_indices = np.hstack(obj_indices) #stack the sequence of input arrays horizontally to make a single array
        _, labels, centers = cv.kmeans(data, b, None, self.criteria, self.atttemps, self.flags) #this is the k-means algorithm
        

        
 
#we generate the vocabulary tree, where data is the SIFT features from the database images, b is the number of branches, dept_tree is the dept of the tree
def hi_kmeans(data, b, dept_tree): #hierarchical k-means algorithm 
    tree = VocabularyTree()
    tree.fit(data, b, dept_tree) #here we fixt the data, the branches and the depth of the tree into the created tree
    nodes = None #saving tree nodes for later use

    return tree

