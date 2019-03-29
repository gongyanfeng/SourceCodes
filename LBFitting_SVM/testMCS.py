#coding=utf-8
import numpy as np
from time import sleep

from sklearn.metrics import roc_auc_score
from sklearn.svm import SVR
from sklearn.externals import joblib
import time
import os
from PIL import Image

INPUT_PATH = 'test_one_MCS'
image_size = 13600 #170*80
image_row = 170    #height
image_column = 80  #width
   
def img2vector(filename):
    returnVect = zeros((1,image_size))
    fr = open(filename)
    for i in range(image_row):
        lineStr = fr.readline()
        for j in range(image_column):
            returnVect[0,image_column*i+j] = int(lineStr[j])
    return returnVect

def loadImages(rawPath):

    rawFileList = os.listdir(rawPath)
    print ("FileList=",rawFileList)
    m = len(rawFileList)
    dataArray = np.zeros((m,image_size))
    labels = []
    for i in range(m):
        fileNameStr = rawFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 1: labels.append(1)
        else: labels.append(-1) 
        print ("fileStr=",fileStr)

        im = Image.open(os.path.join(rawPath,fileNameStr)) 
        im_gray = im.convert('L')
        w = im_gray.size[0] 
        h = im_gray.size[1]
        a = np.zeros((h,w),dtype="uint8")
        im_array = np.array(im_gray)
        for x in range(h):   
            for y in range(w):    
                if im_array[x, y] > 125:
                    a[x][y]= 0
                else:
                    a[x][y] = 1
        dataArray[i,:] = a.reshape(1,13600)
    return dataArray, labels
        
def main(kTup=('rbf', 10)):
    start = time.time()
    dataArr,labelList= loadImages(INPUT_PATH) 
    print("dataArr.shape={},len(labelList)={}".format(dataArr.shape,len(labelList)))
    labelArr = np.array(labelList)

    clf = joblib.load('lbf_svm.joblib')
    test_pred = clf.predict(dataArr)  
    print('Accuracy on test set: {}'.format(roc_auc_score(labelArr, test_pred)))
    end = time.time()
    print("time used is:{}".format((end-start)))

if __name__ == '__main__':
    main(kTup=('rbf', 10))
