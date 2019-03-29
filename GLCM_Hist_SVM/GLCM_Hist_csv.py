#！coding=utf-8
import numpy as np
import os
import pandas as pd
import glcmFeature
import histFeatures


INPUT_PATH='train_and_val' #or 'test'
OUTPUT_PATH='train_and_val.csv'

rawFileList = os.listdir(INPUT_PATH)
#print ("FileList=",rawFileList)
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
    print(glcmhistFeatures)
    glcmhistArray = np.array(glcmhistFeatures).reshape(1,-1)
    print("glcmhistArray.shape()={}".format(glcmhistArray.shape))
    if labelStr=='0':
        target_array=np.array([-1])
    elif labelStr=='1':
        target_array=np.array([1])
    outputs_targets = np.column_stack((glcmhistArray,target_array))#(1,12)glcmhistArray (1,)的targets_array在列上连接形成（1,12+1）维的新array
    print("outputs_targets.shape={}".format(outputs_targets.shape))
    glcmhistArray_list.append(outputs_targets)


bottleneck_array = np.concatenate(glcmhistArray_list)
print('bottleneck_array.shape={}'.format(bottleneck_array.shape))
pd_data=pd.DataFrame(bottleneck_array)
pd_data.to_csv(OUTPUT_PATH,index=False)

