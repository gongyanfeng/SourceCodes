#coding=utf-8
from numpy import *
from time import sleep

from sklearn.metrics import roc_auc_score
from sklearn.svm import SVR
from sklearn.externals import joblib

INPUT_PATH = 'train'
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

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    fileName_afterL = []
    trainingFileList = listdir(dirName) #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,image_size))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 1: hwLabels.append(1)
        else: hwLabels.append(-1)
        fileNameAfter = fileStr.split('_')[1]
        fileName_afterL.append(fileNameAfter)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels, fileName_afterL    

        
def main(kTup=('rbf', 10)):
    dataArr,labelList,fileName_afterL= loadImages(INPUT_PATH) 
    print("dataArr.shape={},len(labelList)={}".format(dataArr.shape,len(labelList)))
    labelArr = array(labelList)

    clf = SVR(gamma='scale',C=1.0,epsilon=0.2)
    clf.fit(dataArr, labelArr)

    file_model = 'lbf_svm.joblib'
    joblib.dump(clf,file_model)

    train_pred = clf.predict(dataArr)
    print('Accuracy on training set: {}'.format(roc_auc_score(labelArr, train_pred)))
    print("train_label.shape={},train_pred.shape={}".format(labelArr.shape,train_pred.shape))

if __name__ == '__main__':
    main(kTup=('rbf', 10))
