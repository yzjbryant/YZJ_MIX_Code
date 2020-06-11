import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.externals import joblib
from sklearn.preprocessing import  OneHotEncoder
from scipy.sparse import hstack
import os
import sys
from pandas import DataFrame
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC, LinearSVR, SVR, NuSVC, NuSVR, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import preprocessing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import json
from xgboost import XGBRegressor
from math import sqrt
from sklearn.model_selection import KFold, cross_val_score
# 分类指标
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 回归指标
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, normalize

data_1 = pd.read_excel('CIF_Parameters.xlsx', header=0)
'''
Feature_Descriptor=data.columns
print(Feature_Descriptor)
'''
Target = ['Pure_C6H6_Heat_298K_9.99kpa',
          'Pure_C6H6_Heat_363K_9.99kpa', 'Pure_C6H6_Heat_363K_99.9kpa',
          'Pure_C6H6_Loading_298K_9.99kpa', 'Pure_C6H6_Loading_363K_9.99kpa',
          'Pure_C6H6_Loading_363K_99.9kpa', 'Pure_C4H4S_Heat_298K_0.01kpa',
          'Pure_C4H4S_Heat_363K_0.01kpa', 'Pure_C4H4S_Heat_363K_0.1kpa',
          'Pure_C4H4S_Loading_298K_0.01kpa', 'Pure_C4H4S_Loading_363K_0.01kpa',
          'Pure_C4H4S_Loading_363K_0.1kpa', 'Mix_Heat_C4H4S_298K_10kpa',
          'Mix_Heat_C4H4S_363K_10kpa', 'Mix_Heat_C4H4S_363K_100kpa',
          'Mix_Heat_C6H6_298K_10kpa', 'Mix_Heat_C6H6_363K_10kpa',
          'Mix_Heat_C6H6_363K_100kpa', 'Mix_Loading_C4H4S_298K_10kpa',
          'Mix_Loading_C4H4S_363K_10kpa', 'Mix_Loading_C4H4S_363K_100kpa',
          'Mix_Loading_C6H6_298K_10kpa', 'Mix_Loading_C6H6_363K_10kpa',
          'Mix_Loading_C6H6_363K_100kpa', 'Selectivity_298K_10kpa',
          'Selectivity_363K_10kpa', 'Selectivity_298K_100kpa',
          'Working_capacity_PSA_363K_100kpa_10kpa',
          'Working_capacity_TSA_10kpa_298K_363K']  # Number of Target(i) is 29
Feature_Descriptor = ['Cluster', 'Linker', 'Group', 'Mass', 'Length_a',
                      'Length_b', 'Length_c', 'Angle_alpha', 'Angle_beta', 'Angle_gamma',
                      'Number of Element', 'Tong', 'Zn', 'C', 'H', 'Br', 'Lu', 'I', 'N', 'O',
                      'F', 'S', 'Di', 'Df', 'Dif', 'Unitcell_volume', 'Density', 'ASA_A^2',
                      'ASA_m^2/cm^3', 'ASA_m^2/g', 'NASA_A^2', 'NASA_m^2/cm^3', 'NASA_m^2/g',
                      'AV_A^3', 'AV_Volume_fraction', 'AV_cm^3/g', 'NAV_A^3',
                      'NAV_Volume_fraction', 'NAV_cm^3/g', 'Pure_C6H6_Heat_298K_9.99kpa',
                      'Pure_C6H6_Heat_363K_9.99kpa', 'Pure_C6H6_Heat_363K_99.9kpa',
                      'Pure_C6H6_Loading_298K_9.99kpa', 'Pure_C6H6_Loading_363K_9.99kpa',
                      'Pure_C6H6_Loading_363K_99.9kpa', 'Pure_C4H4S_Heat_298K_0.01kpa',
                      'Pure_C4H4S_Heat_363K_0.01kpa', 'Pure_C4H4S_Heat_363K_0.1kpa',
                      'Pure_C4H4S_Loading_298K_0.01kpa', 'Pure_C4H4S_Loading_363K_0.01kpa',
                      'Pure_C4H4S_Loading_363K_0.1kpa', 'Mix_Heat_C4H4S_298K_10kpa',
                      'Mix_Heat_C4H4S_363K_10kpa', 'Mix_Heat_C4H4S_363K_100kpa',
                      'Mix_Heat_C6H6_298K_10kpa', 'Mix_Heat_C6H6_363K_10kpa',
                      'Mix_Heat_C6H6_363K_100kpa', 'Mix_Loading_C4H4S_298K_10kpa',
                      'Mix_Loading_C4H4S_363K_10kpa', 'Mix_Loading_C4H4S_363K_100kpa',
                      'Mix_Loading_C6H6_298K_10kpa', 'Mix_Loading_C6H6_363K_10kpa',
                      'Mix_Loading_C6H6_363K_100kpa', 'Selectivity_298K_10kpa',
                      'Selectivity_363K_10kpa', 'Selectivity_298K_100kpa',
                      'Working_capacity_PSA_363K_100kpa_10kpa',
                      'Working_capacity_TSA_10kpa_298K_363K']  # Number of Feature_Descriptor is 69 'Materials',

use_feature = []

data = data_1.drop(['Materials', 'Pure_C6H6_Heat_298K_9.99kpa',
                    'Pure_C6H6_Heat_363K_9.99kpa', 'Pure_C6H6_Heat_363K_99.9kpa',
                    'Pure_C6H6_Loading_298K_9.99kpa', 'Pure_C6H6_Loading_363K_9.99kpa',
                    'Pure_C6H6_Loading_363K_99.9kpa', 'Pure_C4H4S_Heat_298K_0.01kpa',
                    'Pure_C4H4S_Heat_363K_0.01kpa', 'Pure_C4H4S_Heat_363K_0.1kpa',
                    'Pure_C4H4S_Loading_298K_0.01kpa', 'Pure_C4H4S_Loading_363K_0.01kpa',
                    'Pure_C4H4S_Loading_363K_0.1kpa', 'Mix_Heat_C4H4S_298K_10kpa',
                    'Mix_Heat_C4H4S_363K_10kpa', 'Mix_Heat_C4H4S_363K_100kpa',
                    'Mix_Heat_C6H6_298K_10kpa', 'Mix_Heat_C6H6_363K_10kpa',
                    'Mix_Heat_C6H6_363K_100kpa', 'Mix_Loading_C4H4S_298K_10kpa',
                    'Mix_Loading_C4H4S_363K_10kpa', 'Mix_Loading_C4H4S_363K_100kpa',
                    'Mix_Loading_C6H6_298K_10kpa', 'Mix_Loading_C6H6_363K_10kpa',
                    'Mix_Loading_C6H6_363K_100kpa', 'Selectivity_298K_10kpa',
                    'Selectivity_363K_10kpa', 'Selectivity_298K_100kpa',
                    'Working_capacity_PSA_363K_100kpa_10kpa',
                    'Working_capacity_TSA_10kpa_298K_363K'], axis=1)

# Prepare Data
for k in range(len(Target)):
    X = data
    # X=pca.fit_transform(data)
    # X=data.drop(['%s'%Target[k]],axis=1)
    # Feature_Descriptor.remove(Target[k])
    # print(len(Feature_Descriptor))
    Y = data_1['%s' % Target[k]]
    # 标准化
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
    X = scaler.transform(X)
    # Heat、Loading数据采用归一化
    normalizer = preprocessing.Normalizer(copy=True, norm='l2').fit(Y.values.reshape(1, -1))
    Y = normalizer.transform(Y.values.reshape(1, -1))
    Y = np.transpose(Y)
    # Divide Dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=None)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train)
    print(Y_test)
    # 定义模型
    model = xgb.XGBClassifier(nthread=-1,  # 含义：nthread=-1时，使用全部CPU进行并行运算（默认）, nthread=1时，使用1个CPU进行运算。
                              learning_rate=0.08,  # 含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
                              n_estimators=50,  # 含义：总共迭代的次数，即决策树的个数
                              max_depth=5,  # 含义：树的深度，默认值为6，典型值3-10。调参：值越大，越容易过拟合；值越小，越容易欠拟合
                              gamma=0,  # 含义：惩罚项系数，指定节点分裂所需的最小损失函数下降值。
                              subsample=0.9,  # 含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
                              colsample_bytree=0.5)  # 训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
    print("1")
    model.fit(X_train, Y_train)
    print("2")
    # 预测及 AUC 评测
    y_pred_test = model.predict(X_test)#_proba
    print(y_pred_test.shape)
    Y_test = [i for k in Y_test for i in k]
    print(Y_test)
    xgb_test_auc = roc_auc_score(pd.get_dummies(Y_test), y_pred_test.reshape(-1,1))
    print('xgboost test auc: %.5f' % xgb_test_auc)
    xgboost = model
    # xgboost 编码原有特征
    # apply()方法可以获得leaf indices(叶节点索引)
    X_train_leaves = xgboost.apply(X_train)
    # X_train_leaves.shape = (259, 150)
    X_test_leaves = xgboost.apply(X_test)
    # X_test.shape = (112, 4)
    # X_test_leaves.shape = (112, 150)
    # Return the predicted leaf every tree for each sample.


    # 训练样本个数
    train_rows = X_train_leaves.shape[0]
    # 合并编码后的训练数据和测试数据
    X_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)
    X_leaves = X_leaves.astype(np.int32)
    (rows, cols) = X_leaves.shape
    # X_leaves.shape = (371, 150)


    # 对所有特征进行ont-hot编码
    xgbenc = OneHotEncoder()
    X_trans = xgbenc.fit_transform(X_leaves)

    # fit_transform()的作用就是先拟合数据，然后转化它将其转化为标准形式
    # (train_rows, cols) = X_train_leaves.shape

    # 这里得到的X_trans即为得到的one-hot的新特征
    # 定义LR模型
    lr = LogisticRegression()
    # lr对xgboost特征编码后的样本模型训练
    lr.fit(X_trans[:train_rows, :], Y_train)
    y_pred_xgblr1 = lr.predict_proba(X_trans[train_rows:, :])
    #方差
    print(np.var(y_pred_xgblr1))
    # 预测及AUC评测
    # y_pred_xgblr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
    # y_pred_xgblr1.shape  = (112,)
    xgb_lr_auc1 = roc_auc_score(pd.get_dummies(Y_test), y_pred_xgblr1.reshape(-1,1))
    print(Target[k],'基于Xgb特征编码后的LR AUC: %.5f' % xgb_lr_auc1)

    # 将数据分为训练集和测试集进行，用新的特征输入LR进行预测
    # 定义LR模型
    lr = LogisticRegression(n_jobs=-1)
    # 组合特征
    X_train_ext = hstack([X_trans[:train_rows, :], X_train])
    X_test_ext = hstack([X_trans[train_rows:, :], X_test])
    # lr对组合特征的样本模型训练
    lr.fit(X_train_ext,Y_train)
    # 预测及AUC评测
    y_pred_xgblr2 = lr.predict_proba(X_test_ext)
    xgb_lr_auc2 = roc_auc_score(pd.get_dummies(Y_test), y_pred_xgblr2.reshape(-1,1))
    print(Target[k],'基于组合特征的LR AUC: %.5f' % xgb_lr_auc2)