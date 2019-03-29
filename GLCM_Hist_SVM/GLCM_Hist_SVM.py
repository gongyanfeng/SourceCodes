#！coding=utf-8
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.feature_selection import VarianceThreshold
  
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVR
from sklearn.externals import joblib


##########数据准备
df = pd.read_csv("train_and_val.csv")  
print("df.shape={}".format(df.shape)) #<class 'pandas.core.frame.DataFrame'>

#下面返回的train_features等仍是#<class 'pandas.core.frame.DataFrame'>类型
train_features, val_features, train_labels, val_labels = train_test_split(  
    df.drop(labels=['12'], axis=1),
    df['12'],
    test_size=0.0645,
    random_state=41)

######特征自动选择
feature_selector = SequentialFeatureSelector(SVR(gamma='scale',C=1.0,epsilon=0.2),  
           k_features=5,
           forward=True,
           verbose=2,
           scoring='roc_auc',
           cv=4)
features = feature_selector.fit(np.array(train_features.fillna(0)), train_labels)
#将提取到的特征索引保存
print("features.k_feature_idx_={}".format(features.k_feature_idx_))#features.k_feature_idx_)为tuple类型
selected_feature_index_array=np.array(features.k_feature_idx_)
print("selected_feature_index_array={}".format(selected_feature_index_array))
np.savetxt("selected_feature_index.txt",selected_feature_index_array,fmt='%d')

filtered_features= train_features.columns[list(features.k_feature_idx_)]
print("filtered_features.shape={}".format(filtered_features.shape))

######使用分类器对选择的特征训练
#clf = RandomForestClassifier(n_estimators=100, random_state=41, max_depth=3)
clf = SVR(gamma='scale',C=1.0,epsilon=0.2)
clf.fit(train_features[filtered_features].fillna(0), train_labels)

file_model = 'glcm_hist_svm.joblib'
joblib.dump(clf,file_model)

train_pred = clf.predict(train_features[filtered_features].fillna(0))
print('Accuracy on training set: {}'.format(roc_auc_score(train_labels, train_pred)))
print("train_label.shape={},train_pred.shape={}".format(train_labels.shape,train_pred.shape))

val_pred = clf.predict(val_features[filtered_features].fillna(0))  
print('Accuracy on validation set: {}'.format(roc_auc_score(val_labels, val_pred)))