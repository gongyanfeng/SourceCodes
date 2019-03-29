#！coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import gzip
from PIL import Image


multip_matrix = np.matrix([[1, 2, 4], [128, 0,8],[64,32,16]])

def create_lbp_mat(mat):
    
    R,C = mat.shape
    
    transformed = np.zeros((170, 80))
    for i in range (1,R-1):
        for k in range (1,C-1):
            
            binary_matrix = np.zeros((3, 3))
            sub_mat = mat[i-1:i+2,k-1:k+2]
            #print("sub_mat={}".format(sub_mat))
            mid = sub_mat[1,1]
            for t in range(0,3):
                for j in range(0,3):
                    if sub_mat[t,j] > mid:
                        binary_matrix[t,j] = 1
            transformed[i-1,k-1] = np.sum(np.multiply(multip_matrix,binary_matrix))

    return transformed


def getLBF(imgPath):

    LBFeature = []
    img = Image.open(imgPath).convert("L")
    img_array = np.array(img)
    #print("img_array.shape={}".format(img_array.shape))

    res = create_lbp_mat(img_array.reshape((170, 80)))

    unique, counts = np.unique(res, return_counts=True)
    dictionary = dict(zip(unique, counts))
    #print("dictionary={}".format(dictionary))
    for k in range(0,256):
        if k in unique:
            LBFeature.append(dictionary[k])
            
        else:
            LBFeature.append(0)
    #print("len(LBFeature)={}".format(len(LBFeature)))
    return LBFeature
