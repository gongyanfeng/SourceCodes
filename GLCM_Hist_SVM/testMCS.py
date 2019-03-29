#！coding=utf-8
import numpy as np
import os
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
import glcmFeature
import histFeatures
import time

INPUT_PATH='test_one_MCS' #test目录下为经过图像分割后的二值图

start = time.time()
rawFileList = os.listdir(INPUT_PATH)
m = len(rawFileList)
glcmhistArray_list = []
for i in range(m):
    fileNameStr = rawFileList[i]
    fileStr = fileNameStr.split('.')[0]
    labelStr = fileStr.split('_')[0]
    glcmhistFeatures = []
    glcmFeatures_ = list(glcmFeature.getGLCMfeatures(os.path.join(INPUT_PATH,fileNameStr)))
    histFeatures_ = list(histFeatures.getHistfeatures(os.path.join(INPUT_PATH,fileNameStr)))
    for x in glcmFeatures_:
        glcmhistFeatures.append(x[0][0])
    for m in histFeatures_:
        glcmhistFeatures.append(m)
    #print(glcmhistFeatures)
    glcmhistArray = np.array(glcmhistFeatures).reshape(1,-1)
    #print("glcmhistArray.shape()={}".format(glcmhistArray.shape))
    if labelStr=='0':
        target_array=np.array([-1])
    elif labelStr=='1':
        target_array=np.array([1])
    outputs_targets = np.column_stack((glcmhistArray,target_array))#(1,12)glcmhistArray (1,)的targets_array在列上连接形成（1,12+1）维的新array
    #print("outputs_targets.shape={}".format(outputs_targets.shape))
    glcmhistArray_list.append(outputs_targets)

bottleneck_array = np.concatenate(glcmhistArray_list)
test_features=bottleneck_array[:,:-1]
test_labels=bottleneck_array[:,-1]
print("test_features.shape={}".format(test_features.shape))
print("test_labels.shape={}".format(test_labels.shape))

selected_feature_index_array=np.loadtxt("selected_feature_index.txt",dtype= int)
print("selected_feature_index_array={}".format(selected_feature_index_array))
selected_feature_index_list=selected_feature_index_array.tolist()

filtered_features= test_features[:,selected_feature_index_list]
print("filtered_features.shape={}".format(filtered_features.shape))

clf = joblib.load('glcm_hist_svm.joblib')
test_pred = clf.predict(filtered_features)  
print('Accuracy of testing: {}'.format(roc_auc_score(test_labels, test_pred)))
end = time.time()
print("time used is:{}".format((end-start)))
