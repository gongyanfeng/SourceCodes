#！coding=utf-8
import numpy as np
import os
import pandas as pd
import LBF

INPUT_PATH='train_and_val'
OUTPUT_PATH='train_and_val.csv'

rawFileList = os.listdir(INPUT_PATH)
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
    LBF_Array_list.append(outputs_targets)

bottleneck_array = np.concatenate(LBF_Array_list)
print('bottleneck_array.shape={}'.format(bottleneck_array.shape))
pd_data=pd.DataFrame(bottleneck_array)
pd_data.to_csv(OUTPUT_PATH,index=False)
