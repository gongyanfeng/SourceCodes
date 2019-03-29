#！coding=utf-8
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.feature_selection import VarianceThreshold
  
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVR
from sklearn.externals import joblib

CSV_DATASET='train_and_val.csv'
MODEL_OUTPUT='lbf_svm.joblib'
##########数据准备
df = pd.read_csv(CSV_DATASET)  
print("df.shape={}".format(df.shape)) #<class 'pandas.core.frame.DataFrame'>

#下面返回的train_features等仍是#<class 'pandas.core.frame.DataFrame'>类型
train_features, val_features, train_labels, val_labels = train_test_split(  
    df.drop(labels=['256'], axis=1),
    df['256'],
    test_size=0.0645,
    random_state=41)


clf = SVR(gamma='scale',C=1.0,epsilon=0.2)
clf.fit(train_features.fillna(0), train_labels)

joblib.dump(clf,MODEL_OUTPUT)

train_pred = clf.predict(train_features.fillna(0))
print('Accuracy on training set: {}'.format(roc_auc_score(train_labels, train_pred)))
print("train_label.shape={},train_pred.shape={}".format(train_labels.shape,train_pred.shape))

val_pred = clf.predict(val_features.fillna(0))  
print('Accuracy on validation set: {}'.format(roc_auc_score(val_labels, val_pred)))