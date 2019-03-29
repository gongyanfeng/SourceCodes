#! /usr/bin/env python
#coding=utf-8
from numpy import *
import os.path
import glob
import tensorflow as tf

INPUT_DATA = 'bottleneckBullet/'
GOOD_DIR = 'bottleneckBullet/good'
BAD_DIR = 'bottleneckBullet/bad'
CLASS_GOOD = 'good'
CLASS_BAD = 'bad'
bottleneck_size = 2048

VALIDATION_NUM_BAD = 50
VALIDATION_NUM_GOOD = 50

SUMMARY_DIR_VAL = 'log/val'
weights_path = "weights.txt"  #用于存储训练结果（训练结果仅此一项）

def create_bottleneck_lists():
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        # 获取当前目录下所有的有效bottleneck文件。       
        file_list = []
        #dir_name：bad或good
        dir_name = os.path.basename(sub_dir)
        file_glob = os.path.join(INPUT_DATA, dir_name, '*.txt')
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        # 通过目录名获取类别的名称。
        label_name = dir_name.lower()
        # 初始化当前类别的训练数据集、测试数据集和验证数据集
        training_bottlenecks = []
        validation_bottlenecks = []
        bottlenecks_list = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            bottlenecks_list.append(base_name)
        #因为sub_dirs下有两个子目录Tshape和Xshape,故如下打印('validation_images[]=', ['026.jpg', '015.jpg', '005.jpg'])
        #        ('validation_images[]=', ['030.jpg', '027.jpg'])
        if dir_name == CLASS_GOOD: #'good'
            validation_bottlenecks = bottlenecks_list[:VALIDATION_NUM_GOOD]
            training_bottlenecks = bottlenecks_list[VALIDATION_NUM_GOOD:]
        elif dir_name == CLASS_BAD: #'bad'
            validation_bottlenecks = bottlenecks_list[:VALIDATION_NUM_BAD]
            training_bottlenecks = bottlenecks_list[VALIDATION_NUM_BAD:]
        # 将当前类别的数据放入结果字典。
        result[label_name] = {
            'dir': dir_name,
            'training': training_bottlenecks,
            'validation': validation_bottlenecks,
            }
    # 返回整理好的所有数据
    #print(result)
    return result

# 这个函数获取全部的测试bottleneck。在最终测试的时候需要在所有的测试数据上计算正确率。
def get_validation_bottlenecks(bottleneck_lists, n_classes):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(bottleneck_lists.keys())
    # 枚举所有的类别和每个类别中的测试图片。
    for label_index, label_name in enumerate(label_name_list):
        category = 'validation'       
        for index, unused_base_name in enumerate(bottleneck_lists[label_name][category]):
            # 通过Inception-v3模型计算图片对应的特征向量，并将其加入最终数据的列表。
            bottleneck, _= get_bottleneck(bottleneck_lists, label_name, index, category)
            if label_name == CLASS_BAD: #'bad'
                ground_truths.append(0)
            elif label_name == CLASS_GOOD: # 'good'
                ground_truths.append(1)
            bottlenecks.append(bottleneck)
    return bottlenecks, ground_truths

# 这个函数获取全部的训练bottleneck。
def get_train_bottlenecks(bottleneck_lists, n_classes):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(bottleneck_lists.keys())
    # 枚举所有的类别和每个类别中的测试图片。
    for label_index, label_name in enumerate(label_name_list):
        category = 'training'       
        for index, unused_base_name in enumerate(bottleneck_lists[label_name][category]):
            # 通过Inception-v3模型计算图片对应的特征向量，并将其加入最终数据的列表。
            bottleneck, _ = get_bottleneck(bottleneck_lists, label_name, index, category)
            if label_name == CLASS_BAD: #'bad'
                ground_truths.append(0)
            elif label_name == CLASS_GOOD: # 'good'
                ground_truths.append(1)
            bottlenecks.append(bottleneck)
    return bottlenecks, ground_truths

def get_bottleneck(bottleneck_lists, label_name, index, category):
    # 获取一张图片对应的特征向量文件的路径。
    label_lists = bottleneck_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(INPUT_DATA, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path, base_name = get_bottleneck_path(bottleneck_lists, INPUT_DATA, label_name, index, category)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    # 返回得到的特征向量
    return bottleneck_values, base_name

# 这个函数通过类别名称、所属数据集和图片编号获取经过Inception-v3模型处理之后的特征向量文件地址
def get_bottleneck_path(bottleneck_lists, bottleneck_dir, label_name, index, category):
    label_lists = bottleneck_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(bottleneck_dir, sub_dir, base_name)
    return full_path,base_name

def sigmoid(inX):
    return 1.0/(1+exp(-longfloat(inX)))  #Runtime Warning: overflow encountered in exp


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    #if prob > 0.1: return 1.0
    else: return 0.0

def writeToTxt(im_array,file_path):
    try:
        fp = open(file_path,"w+")
        im_array_h = im_array.shape[0]
        for i in range(im_array_h):
            fp.write(str(im_array[i])+' ')
            #fp.write("\n")
        fp.close()
    except IOError:
        print("fail to open file")
    
def main():

    true_bad = tf.placeholder(tf.float32, shape=None, name='true_bad_Placeholder')
    true_good = tf.placeholder(tf.float32, shape=None, name='true_good_Placeholder')
    false_bad = tf.placeholder(tf.float32, shape=None, name='false_bad_Placeholder')
    false_good =tf.placeholder(tf.float32, shape=None, name='false_good_Placeholder')
    rightCount = tf.placeholder(tf.float32, shape=None, name='false_good_Placeholder')
    totalNum = tf.placeholder(tf.float32, shape=None, name='false_good_Placeholder')
    
    if true_bad+false_bad==0: precision=0
    else: precision = (true_bad)/(true_bad+false_bad)
    #tf.scalar_summary('precision',precision)
    summary_precision = tf.summary.scalar('precision',precision)
    if true_bad+false_good==0: recall=0
    else: recall = (true_bad)/(true_bad+false_good)
    summary_recall = tf.summary.scalar('recall',recall)
    accuracy = (true_bad+true_good)/(true_bad+true_good+false_bad+false_good)
    summary_accuracy = tf.summary.scalar('accuracy',accuracy)
    if true_bad+false_bad+false_good==0: f1_measure=0
    else: f1_measure = (2*true_bad)/(2*true_bad+false_bad+false_good)
    summary_f1 = tf.summary.scalar('f1_measure',f1_measure)
    total_accuracy = rightCount/totalNum
    summary_total_accuracy = tf.summary.scalar('total_accuracy',total_accuracy)
    
    bottleneck_lists = create_bottleneck_lists()
    n_classes = len(bottleneck_lists.keys())
    
    trainingSet,trainingLabels = get_train_bottlenecks(bottleneck_lists, n_classes)
    validationSet,validationLabels= get_validation_bottlenecks(bottleneck_lists, n_classes)
    
    n_validation = len(validationSet)
    
    if trainingLabels:
        
        merged_val = tf.summary.merge([summary_precision,summary_recall,summary_accuracy,summary_f1,summary_total_accuracy])
            #merged_test = tf.summary.merge([summary_precision,summary_recall,summary_accuracy,summary_f1])
        with tf.Session() as sess:
            summary_writer_val = tf.summary.FileWriter(SUMMARY_DIR_VAL,sess.graph)
            #tf.global_variables_initializer().run()
            tf.initialize_all_variables().run()
            
            #训练用
            dataMatrix = array(trainingSet)
            m,n = shape(dataMatrix) #m=1749,n=2048
            print ('m=',m)
            print ('n=',n)
            Weights = ones(n)   #initialize to all ones
            for j in range(1000):
                dataIndex = range(m)
                #print ('dataIndex=',dataIndex)
                for i in range(m):
                    alpha = 4/(1.0+j+i)+0.01    #apha decreases with iteration, does not 
                    randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
                    h = sigmoid(sum(dataMatrix[randIndex]*Weights))
                    error = trainingLabels[randIndex] - h
                    Weights = Weights + alpha * error * dataMatrix[randIndex]
                    del(dataIndex[randIndex])
                
                
                #下面利用训练好的参数trainWeights计算训练数据集的精度 
                print ('iteration step j=',j)
                print ('validationSet.len=',n_validation)
                true_bad_ = 0
                true_good_ = 0
                false_bad_ = 0
                false_good_ = 0
                rightCount_ = 0
                               
                for i_validation in range(n_validation):        
                    pred_result = int(classifyVector(validationSet[i_validation], Weights))
                    label = int(validationLabels[i_validation])
                    if pred_result==0 and label==0: true_bad_ +=1
                    elif pred_result==1 and label==1: true_good_ +=1
                    elif pred_result==0 and label==1:
                        false_bad_ +=1
                    elif pred_result==1 and label==0:
                        false_good_ +=1
                    if int(classifyVector(validationSet[i_validation], Weights)) == int(validationLabels[i_validation]):
                        rightCount_ += 1
                
                print 'true_bad= %d' % (true_bad_)
                print 'true_good= %d' % (true_good_)
                print 'false_bad= %d' % (false_bad_)
                print 'false_good= %d' % (false_good_)    
                print ('rightCount_=',rightCount_)
                summary_val,precision_,recall_,accuracy_,f1_measure_,total_accuracy_= sess.run(
                    [merged_val,precision,recall,accuracy,f1_measure,total_accuracy],
                    feed_dict={true_bad:true_bad_, true_good:true_good_, false_bad:false_bad_, false_good:false_good_,rightCount:rightCount_,totalNum:n_validation})
                
                print ('precision=',precision_)
                print ('recall=',recall_)
                print ('accuracy=',accuracy_)
                print ('f1_measure=',f1_measure_)
                print ('total_accuracy_=',total_accuracy_)

                summary_writer_val.add_summary(summary_val,j)
            
            
        writeToTxt(Weights,weights_path)
    else:
        print "*** the trainingLabels is empty, please check the path of INPUT_DATA path"

if __name__ == '__main__':
    main()


