# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:26:18 2017

@author: Mohammad Imtiaz
"""

import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
#from skimage import data

def getGLCMfeatures(img_path):
    img = cv2.imread(img_path)
    img = img[:,:,0]

    glcm = greycomatrix(img, [1], [0],  symmetric = True, normed = True )
    contrast = greycoprops(glcm, 'contrast')
    dissimilarityraster = greycoprops(glcm, 'dissimilarity')
    homogeneityraster = greycoprops(glcm, 'homogeneity')
    energyraster = greycoprops(glcm, 'energy')
    correlationraster = greycoprops(glcm, 'correlation')
    ASMraster = greycoprops(glcm, 'ASM')
    '''
    print("contrast={}".format(contrast))
    print("dissimilarityraster={}".format(dissimilarityraster))
    print("homogeneityraster={}".format(homogeneityraster))
    print("energyraster={}".format(energyraster))
    print("correlationraster={}".format(correlationraster))
    print("ASMraster={}".format(ASMraster))
    '''
    return ASMraster,contrast,correlationraster,homogeneityraster,energyraster,dissimilarityraster







