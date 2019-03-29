#! /usr/bin/env python
#coding=utf-8

from numpy import *
import os.path
import glob
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import time
import cv2
import serial


INPUT_DATA= 'cap_images/'
CACHE_DIR = 'cap_images_bottleneck/'

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
# 图像输入张量所对应的名称。
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
MODEL_DIR = 'inception_dec_2015/'
MODEL_FILE = 'tensorflow_inception_graph.pb'
weights_path = "weights.txt"

def create_image_lists():
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(INPUT_DATA, '*.'+extension)
        file_list.extend(glob.glob(file_glob))
    return file_list

def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量（一维数组）
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def create_bottleneck(sess, image_path, jpeg_data_tensor, bottleneck_tensor):
    base_name = os.path.basename(image_path)
    bottleneck_path = os.path.join(CACHE_DIR, base_name) + '.txt'
    image_data = gfile.FastGFile(image_path, 'rb').read()
    # 通过Inception-v3模型计算特征向量
    bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
    # 将计算得到的特征向量存入文件
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)   
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)   
    # 返回得到的特征向量
    return bottleneck_values

def get_test_bottlenecks(sess, images_list,jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    fileNameList = []
    for image_path in images_list:
        bottleneck = create_bottleneck(sess, image_path,jpeg_data_tensor, bottleneck_tensor)
        bottlenecks.append(bottleneck)
        
        base_name = os.path.basename(image_path)
        fileStr = base_name.split('.')[0]     #take off .jpg.txt get "*_*"
        fileNameList.append(fileStr)
        
    print ('creating_test_bottlenecks finished!')
    return bottlenecks,fileNameList

def sigmoid(inX):
    return 1.0/(1+exp(-longfloat(inX)))  #Runtime Warning: overflow encountered in exp

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def readFromTxt(txtFileName,row,column):
    a = loadtxt(txtFileName) #a的类型为numpy.ndarray,shape是一维的:(2048,)，其元素a[1]类型为numpy.float64
    #print 'shape(a)=',shape(a) #一维的，（2048,0）,所以需要reshape成（2048,1）
    return a

def Test(sess, images_list,jpeg_data_tensor, bottleneck_tensor):
    testSet,fileNameList = get_test_bottlenecks(sess, images_list,jpeg_data_tensor, bottleneck_tensor)          
    trainWeights = readFromTxt(weights_path,2048,1)       
    #下面利用训练好的参数trainWeights计算训练数据集的精度
    errorCount = 0; numTestVec = 0.0
    n = len(testSet)
    print ('The number of testImages is ={}'.format(n))
    true_bad = 0
    true_good = 0
    false_bad = 0
    false_good = 0
    total_results = []
    for i in range(n):
        numTestVec += 1.0
        pred_result = int(classifyVector(testSet[i], trainWeights))
        total_results.append(pred_result)
        print ('pred_result={}'.format(pred_result))
        if pred_result==0:
            res = 'Defective'
        elif pred_result==1:
            res = 'Non-defective'
        print ('fileName:{},detection result={}'.format(fileNameList[i],res))
    
    results_sum = sum(total_results)
    if results_sum!=5:
        print("This MCS has defects.")
        
def main():
    start =time.clock() #开始计时
    
    rawFileList = os.listdir(INPUT_DATA)
    m = len(rawFileList)
    if m!=5:         
        ser = serial.Serial('/dev/ttyUSB0',9600,timeout=0.5)
        print ("the information of serial：",ser)
        print ("myserial is open?：",ser.isOpen)    
        base_path="cap_images/"
        cap = cv2.VideoCapture(0)   
        success, frame = cap.read()   
        frame_num = 0
        num = 21
        start = time.clock()
        while(success and frame_num<=num):
            success, frame = cap.read()	
            if(frame_num>4 and frame_num % 4 ==1):
                file_name=os.path.join(base_path,str(frame_num)+'.jpg') #会保存第5,9,13,17，21帧
                frame = frame[60:450,200:430] #(230,400)
                frame = cv2.resize(frame,(80,170),interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(file_name, frame)
                print ("frame_num={}".format(frame_num))
                ser.write("\x42")		
                
            frame_num += 1
            cv2.waitKey(1)       
        end = time.clock()
        print ("time used ={}".format(end*1000-start*1000))   
        cap.release()
        cv2.destroyAllWindows()
    
    
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)      

    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # 加载读取的Inception-v3模型，并返回数据输入所对应的张量以及计算瓶颈层结果所对应的张量。
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
       
    images_list = []
    images_list = create_image_lists()
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    
    with tf.Session() as sess:        
        tf.global_variables_initializer().run()
        #tf.initialize_all_variables().run()
        Test(sess, images_list,jpeg_data_tensor, bottleneck_tensor)
    end = time.clock()
    print('Running time: %s Seconds'%(end-start))

if __name__ == '__main__':
    main()
