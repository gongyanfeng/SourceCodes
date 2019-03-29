#！coding=utf-8
import numpy as np
import os
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
import LBF
import time


INPUT_PATH='test_one_MCS'
MODEL_PATH='lbf_svm.joblib'

start = time.time()
rawFileList = os.listdir(INPUT_PATH)
#print ("FileList=",rawFileList)
m = len(rawFileList)
LBF_Array_list = []
for i in range(m):
    fileNameStr = rawFileList[i]
    fileStr = fileNameStr.split('.')[0]
    labelStr = fileStr.split('_')[0]
    glcmhistFeatures = []
    LBF_list=LBF.getLBF(os.path.join(INPUT_PATH,fileNameStr))
    LBF_Array = np.array(LBF_list).reshape(1,-1)
    if labelStr=='0':
        target_array=np.array([-1])
    elif labelStr=='1':
        target_array=np.array([1])
    outputs_targets = np.column_stack((LBF_Array,target_array))
    #print("outputs_targets.shape={}".format(outputs_targets.shape))
    LBF_Array_list.append(outputs_targets)


bottleneck_array = np.concatenate(LBF_Array_list)
print('bottleneck_array.shape={}'.format(bottleneck_array.shape))
test_features=bottleneck_array[:,:-1]
test_labels=bottleneck_array[:,-1]
print("test_features.shape={}".format(test_features.shape))
print("test_labels.shape={}".format(test_labels.shape))

clf = joblib.load(MODEL_PATH)
test_pred = clf.predict(test_features)  
print('Accuracy of testing: {}'.format(roc_auc_score(test_labels, test_pred)))
end = time.time()
print("time used is:{}".format((end-start)))
