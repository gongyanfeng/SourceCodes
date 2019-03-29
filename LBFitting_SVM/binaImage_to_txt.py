#! /usr/bin/env python
#coding=utf-8
from PIL import Image
import numpy as np

import os

#rawPath = 'bina_good'
#txtPath = "txt_good/"
rawPath = 'bina_bad'
txtPath = "txt_bad/"

def writeToTxt(im_array,file_path):
    try:
        fp = open(file_path,"w+")
        im_array_h = im_array.shape[0]
        for i in range(im_array_h):
            for item in im_array[i]:
                fp.write(str(item))
            fp.write("\n")
        fp.close()
    except IOError:
        print("fail to open file")

def main():
    rawFileList = os.listdir(rawPath)
    print ("FileList=",rawFileList)
    m = len(rawFileList)
    for i in range(m):
        fileNameStr = rawFileList[i]
        fileStr = fileNameStr.split('.')[0]
        #print ("fileNameStr=",fileNameStr)
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
        file_path = os.path.join(txtPath,fileStr+".txt")
        writeToTxt(a,file_path)
        
if __name__ == "__main__":
    main()