import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostClassifier,AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC,LinearSVR,SVR,NuSVC,NuSVR,OneClassSVM
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn import preprocessing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import Ridge,Lasso,ElasticNet,LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import json
import xgboost as xgb
from xgboost import XGBRegressor
from math import sqrt
#分类指标
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#回归指标
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#导入数据
data=pd.read_excel('CIF_Parameters.xlsx',header=0)
pd.set_option('display.max_rows', 6000)
pd.set_option('display.max_columns', 6000)
pd.set_option('display.width', 1000)

#
# print(data)
# print("data============================================")
# print(data.info())
# print("data.info()============================================")
# print(data.head())
# print("data.head()============================================")
# print(data.describe())
# print("data.describe()============================================")
# print(data.dtypes)
# print("data.dtypes============================================")
# 处理各字段
# X_train=data.drop(["Materials"],axis=1)
# Y_train=data["Energy_C4H4S_298K_10kpa"]
# X_test=data_test.drop(["Materials","Energy_C4H4S_298K_10kpa"],axis=1).copy()

'''
#归一化  无用
# scaler=Normalizer().fit(X_train)
# X_train=scaler.transform(X_train)
# X_test=scaler.transform(X_test)
'''

# Energy_C4H4S_298K_10kpa	Energy_C4H4S_363K_10kpa	Energy_C4H4S_363K_100kpa
# Energy_C6H6_298K_10kpa	Energy_C6H6_363K_10kpa	Energy_C6H6_363K_100kpa
# Loading_C4H4S_298K_10kpa	Loading_C4H4S_363K_10kpa Loading_C4H4S_363K_100kpa
# Loading_C6H6_298K_10kpa	Loading_C6H6_363K_10kpa	Loading_C6H6_363K_100kpa

Target=['Loading_C4H4S_298K_10kpa','Loading_C4H4S_363K_10kpa','Loading_C4H4S_363K_100kpa',
        'Loading_C6H6_298K_10kpa','Loading_C6H6_363K_10kpa','Loading_C6H6_363K_100kpa']
Feature_Descriptor=['Materials','Cluster','Linker','Group','Length_a','Length_b','Length_c','Angle_alpha','Angle_beta',
                    'Angle_gamma','Number of Element','Tong','Zn','C','H','Br','Lu','I','N','O','F','S','Di','Df','Dif',
                    'Unitcell_volume','Density','ASA_A^2','ASA_m^2/cm^3','ASA_m^2/g','NASA_A^2','NASA_m^2/cm^3',
                    'NASA_m^2/g','AV_A^3','AV_Volume_fraction','AV_cm^3/g','NAV_A^3','NAV_Volume_fraction',
                    'NAV_cm^3/g','Energy_C4H4S_298K_10kpa','Energy_C4H4S_363K_10kpa','Energy_C4H4S_363K_100kpa',
                    'Energy_C6H6_298K_10kpa','Energy_C6H6_363K_10kpa','Energy_C6H6_363K_100kpa','Loading_C4H4S_298K_10kpa',
                    'Loading_C4H4S_363K_10kpa','Loading_C4H4S_363K_100kpa','Loading_C6H6_298K_10kpa','Loading_C6H6_363K_10kpa',
                    'Loading_C6H6_363K_100kpa']

X=data.drop(['Materials','Energy_C4H4S_298K_10kpa','Energy_C4H4S_363K_10kpa','Energy_C4H4S_363K_100kpa',
                    'Energy_C6H6_298K_10kpa','Energy_C6H6_363K_10kpa','Energy_C6H6_363K_100kpa',
             'Loading_C4H4S_298K_10kpa'],axis=1)
Y=data['Loading_C4H4S_298K_10kpa']
#数据可视化
correlations=X.corr()
correction=abs(correlations)
fig=plt.figure()
ax=fig.add_subplot(figsize=(20,20))
ax1=sns.heatmap(correction) #相关性热力图
#定义训练集与测试集
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=None)
# print(X_train,X_test,Y_train,Y_test)


#数据预处理
#标准化
scaler=StandardScaler(copy=True,with_mean=True,with_std=True).fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#############################################################################################建模分析
#file:///D:/thesis/Python/Python书籍/书籍/1/AI%20算法工程师手册/工具/scikit-learn/3.supervised_model.html

Machine_Learning_Results={}

#################################################################################################
#一、线性模型
#1.1 LinearRegression-线性回归模型
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
for i in range(10):
    linear=LinearRegression(fit_intercept=True,normalize=False,
                            copy_X=True,n_jobs=1)
    linear.fit(X_train,Y_train)
    Y_pred=linear.predict(X_test)
    # 计算得分
    a=linear.score(X_train,Y_train)
    b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train = list(Y_train)
    Y_test = list(Y_test)
    Y_pred = list(Y_pred)
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)
    Y_test_mean = np.mean(Y_test)
    Y_pred_mean = np.mean(Y_pred)

    # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
    def rmse(Y_test, Y_pred):  # RMSE
        error = []
        for i in range(len(Y_test)):
            error.append(Y_test[i] - Y_pred[i])
        squareError = []
        for i in error:
            squareError.append(i * i)
        return sqrt(sum(squareError) / len(squareError))
    c = mean_absolute_error(Y_test, Y_pred)
    d = mean_squared_error(Y_test, Y_pred)
    e = rmse(Y_test, Y_pred)
    f = r2_score(Y_test, Y_pred)
    # MAPE
    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            Y_test[i] = Y_test_mean
    for i in range(len(Y_pred)):
        if Y_pred[i] == 0:
            Y_pred[i] = Y_pred_mean
    def mape(Y_test, Y_pred):
        n = len(Y_test)
        mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
        return mape
    g = mape(Y_test, Y_pred)
    MAE.append(c)
    MSE.append(d)
    RMSE.append(e)
    R2.append(f)
    MAPE.append(g)
    # 绘图
    fig = plt.figure()
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    # #plt.show()
print("LinearRegression:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
      np.mean(MAPE))


#1.2 Ridge-岭回归
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
for i in range(10):
    ridge = Ridge(alpha=1.0, fit_intercept=True, normalize=False,
                  copy_X=True, max_iter=None, tol=0.001,
                  solver='auto', random_state=None)
    ridge.fit(X_train,Y_train)
    Y_pred=ridge.predict(X_test)
    # 计算得分
    a=ridge.score(X_train,Y_train)
    b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train = list(Y_train)
    Y_test = list(Y_test)
    Y_pred = list(Y_pred)
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)
    Y_test_mean = np.mean(Y_test)
    Y_pred_mean = np.mean(Y_pred)

    # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
    def rmse(Y_test, Y_pred):  # RMSE
        error = []
        for i in range(len(Y_test)):
            error.append(Y_test[i] - Y_pred[i])
        squareError = []
        for i in error:
            squareError.append(i * i)
        return sqrt(sum(squareError) / len(squareError))
    c = mean_absolute_error(Y_test, Y_pred)
    d = mean_squared_error(Y_test, Y_pred)
    e = rmse(Y_test, Y_pred)
    f = r2_score(Y_test, Y_pred)
    # MAPE
    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            Y_test[i] = Y_test_mean
    for i in range(len(Y_pred)):
        if Y_pred[i] == 0:
            Y_pred[i] = Y_pred_mean
    def mape(Y_test, Y_pred):
        n = len(Y_test)
        mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
        return mape
    g = mape(Y_test, Y_pred)
    MAE.append(c)
    MSE.append(d)
    RMSE.append(e)
    R2.append(f)
    MAPE.append(g)
    # 绘图
    fig = plt.figure()
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    # #plt.show()
print("Ridge:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
      np.mean(MAPE))


#1.3 Lasso-Lasso回归
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
for i in range(10):
    lasso = Lasso(alpha=1.0, fit_intercept=True, normalize=False,
                  copy_X=True, max_iter=1000, tol=0.0001, warm_start=False,
                  positive=False, random_state=None, selection='cyclic')
    lasso.fit(X_train,Y_train)
    Y_pred=lasso.predict(X_test)
    # 计算得分
    a=lasso.score(X_train,Y_train)
    b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train = list(Y_train)
    Y_test = list(Y_test)
    Y_pred = list(Y_pred)
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)
    Y_test_mean = np.mean(Y_test)
    Y_pred_mean = np.mean(Y_pred)

    # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
    def rmse(Y_test, Y_pred):  # RMSE
        error = []
        for i in range(len(Y_test)):
            error.append(Y_test[i] - Y_pred[i])
        squareError = []
        for i in error:
            squareError.append(i * i)
        return sqrt(sum(squareError) / len(squareError))
    c = mean_absolute_error(Y_test, Y_pred)
    d = mean_squared_error(Y_test, Y_pred)
    e = rmse(Y_test, Y_pred)
    f = r2_score(Y_test, Y_pred)
    # MAPE
    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            Y_test[i] = Y_test_mean
    for i in range(len(Y_pred)):
        if Y_pred[i] == 0:
            Y_pred[i] = Y_pred_mean
    def mape(Y_test, Y_pred):
        n = len(Y_test)
        mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
        return mape
    g = mape(Y_test, Y_pred)
    MAE.append(c)
    MSE.append(d)
    RMSE.append(e)
    R2.append(f)
    MAPE.append(g)
    # 绘图
    fig = plt.figure()
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    # #plt.show()
print("Lasso:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
      np.mean(MAPE))


#1.4 ElasticNet-ElasticNet回归
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
for i in range(10):
    elasticnet = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                            normalize=False, copy_X=True, max_iter=1000,
                            tol=0.0001, warm_start=False, precompute=False,
                            positive=False, random_state=None,
                            selection='cyclic')
    elasticnet.fit(X_train,Y_train)
    Y_pred=elasticnet.predict(X_test)
    # 计算得分
    a=elasticnet.score(X_train,Y_train)
    b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train = list(Y_train)
    Y_test = list(Y_test)
    Y_pred = list(Y_pred)
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)
    Y_test_mean = np.mean(Y_test)
    Y_pred_mean = np.mean(Y_pred)

    # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
    def rmse(Y_test, Y_pred):  # RMSE
        error = []
        for i in range(len(Y_test)):
            error.append(Y_test[i] - Y_pred[i])
        squareError = []
        for i in error:
            squareError.append(i * i)
        return sqrt(sum(squareError) / len(squareError))
    c = mean_absolute_error(Y_test, Y_pred)
    d = mean_squared_error(Y_test, Y_pred)
    e = rmse(Y_test, Y_pred)
    f = r2_score(Y_test, Y_pred)
    # MAPE
    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            Y_test[i] = Y_test_mean
    for i in range(len(Y_pred)):
        if Y_pred[i] == 0:
            Y_pred[i] = Y_pred_mean
    def mape(Y_test, Y_pred):
        n = len(Y_test)
        mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
        return mape
    g = mape(Y_test, Y_pred)
    MAE.append(c)
    MSE.append(d)
    RMSE.append(e)
    R2.append(f)
    MAPE.append(g)
    # 绘图
    fig = plt.figure()
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    # #plt.show()
print("ElasticNet:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
      np.mean(MAPE))


#1.5 LogisticRegression-对数几率回归模型
# MAE=[]
# MSE=[]
# RMSE=[]
# R2=[]
# MAPE=[]
# for i in range(10):
#     logreg = LogisticRegression(penalty='l2', dual=False, tol=0.0001,
#                                 C=1.0, fit_intercept=True, intercept_scaling=1,
#                                 class_weight=None, random_state=None,
#                                 solver='liblinear', max_iter=100, multi_class='ovr',
#                                 verbose=0, warm_start=False, n_jobs=1)
#     logreg.fit(X_train.astype('int'),Y_train.astype('int'))
#     Y_pred=logreg.predict(X_test)
#     # 计算得分
#     a=logreg.score(X_train,Y_train)
#     b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
#     Y_train = list(Y_train)
#     Y_test = list(Y_test)
#     Y_pred = list(Y_pred)
#     Y_test = np.array(Y_test)
#     Y_pred = np.array(Y_pred)
#     Y_test_mean = np.mean(Y_test)
#     Y_pred_mean = np.mean(Y_pred)
#
#     # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
#     # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
#     # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
#     def rmse(Y_test, Y_pred):  # RMSE
#         error = []
#         for i in range(len(Y_test)):
#             error.append(Y_test[i] - Y_pred[i])
#         squareError = []
#         for i in error:
#             squareError.append(i * i)
#         return sqrt(sum(squareError) / len(squareError))
#     c = mean_absolute_error(Y_test, Y_pred)
#     d = mean_squared_error(Y_test, Y_pred)
#     e = rmse(Y_test, Y_pred)
#     f = r2_score(Y_test, Y_pred)
#     # MAPE
#     for i in range(len(Y_test)):
#         if Y_test[i] == 0:
#             Y_test[i] = Y_test_mean
#     for i in range(len(Y_pred)):
#         if Y_pred[i] == 0:
#             Y_pred[i] = Y_pred_mean
#     def mape(Y_test, Y_pred):
#         n = len(Y_test)
#         mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
#         return mape
#     g = mape(Y_test, Y_pred)
#     MAE.append(c)
#     MSE.append(d)
#     RMSE.append(e)
#     R2.append(f)
#     MAPE.append(g)
#     # 绘图
#     fig = plt.figure()
#     plt.scatter(Y_test, Y_pred)
#     plt.xlabel("Y_test")
#     plt.ylabel("Y_pred")
#     plt.xlim(0, 20)
#     plt.ylim(0, 20)
#     # #plt.show()
# print("LogisticRegression:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
#       np.mean(MAPE))


#1.6 LinearDiscriminantAnalysis-线性判别分析模型
# lineardisana=LinearDiscriminantAnalysis(solver='svd',shrinkage=None,
#                                         priors=None,n_components=None,
#                                         store_covariance=False,tol=0.0001)
#                     #option of solver : svd,lsqr,eigen
# lineardisana.fit(X_train.astype('int'),Y_train.astype('int'))
# Y_pred=lineardisana.predict(X_test.astype('int'))
# a=lineardisana.score(X_train.astype('int'),Y_train.astype('int'))
# b=lineardisana.predict_log_proba(X_train)
# c=lineardisana.predict_proba(X_train)
# print("LinearDiscrimiantAnalysis:",a,b,c)
'''

'''
#################################################################################LinearSVC
#二、支持向量机
#2.1 LinearSVC
# MAE=[]
# MSE=[]
# RMSE=[]
# R2=[]
# MAPE=[]
# for i in range(10):
#     svc = LinearSVC(penalty='l2', loss='squared_hinge', dual=True,
#                     tol=0.0001, C=1.0, multi_class='ovr',
#                     fit_intercept=True, intercept_scaling=1,
#                     class_weight=None, verbose=0, random_state=None,
#                     max_iter=1000)
#     svc.fit(X_train,Y_train)
#     Y_pred=svc.predict(X_test)
#     # 计算得分
#     a=svc.score(X_train,Y_train)
#     b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
#     Y_train = list(Y_train)
#     Y_test = list(Y_test)
#     Y_pred = list(Y_pred)
#     Y_test = np.array(Y_test)
#     Y_pred = np.array(Y_pred)
#     Y_test_mean = np.mean(Y_test)
#     Y_pred_mean = np.mean(Y_pred)
#
#     # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
#     # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
#     # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
#     def rmse(Y_test, Y_pred):  # RMSE
#         error = []
#         for i in range(len(Y_test)):
#             error.append(Y_test[i] - Y_pred[i])
#         squareError = []
#         for i in error:
#             squareError.append(i * i)
#         return sqrt(sum(squareError) / len(squareError))
#     c = mean_absolute_error(Y_test, Y_pred)
#     d = mean_squared_error(Y_test, Y_pred)
#     e = rmse(Y_test, Y_pred)
#     f = r2_score(Y_test, Y_pred)
#     # MAPE
#     for i in range(len(Y_test)):
#         if Y_test[i] == 0:
#             Y_test[i] = Y_test_mean
#     for i in range(len(Y_pred)):
#         if Y_pred[i] == 0:
#             Y_pred[i] = Y_pred_mean
#     def mape(Y_test, Y_pred):
#         n = len(Y_test)
#         mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
#         return mape
#     g = mape(Y_test, Y_pred)
#     MAE.append(c)
#     MSE.append(d)
#     RMSE.append(e)
#     R2.append(f)
#     MAPE.append(g)
#     # 绘图
#     fig = plt.figure()
#     plt.scatter(Y_test, Y_pred)
#     plt.xlabel("Y_test")
#     plt.ylabel("Y_pred")
#     plt.xlim(0, 20)
#     plt.ylim(0, 20)
#     # #plt.show()
# print("LinearSVC:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
#       np.mean(MAPE))


#2.2 SVC
# MAE=[]
# MSE=[]
# RMSE=[]
# R2=[]
# MAPE=[]
# for i in range(10):
#     svc = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
#               shrinking=True, probability=False, tol=0.001, cache_size=200,
#               class_weight=None, verbose=False, max_iter=-1,
#               decision_function_shape=None, random_state=None)
#     svc.fit(X_train,Y_train)
#     Y_pred=svc.predict(X_test)
#     # 计算得分
#     a=svc.score(X_train,Y_train)
#     b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
#     Y_train = list(Y_train)
#     Y_test = list(Y_test)
#     Y_pred = list(Y_pred)
#     Y_test = np.array(Y_test)
#     Y_pred = np.array(Y_pred)
#     Y_test_mean = np.mean(Y_test)
#     Y_pred_mean = np.mean(Y_pred)
#
#     # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
#     # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
#     # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
#     def rmse(Y_test, Y_pred):  # RMSE
#         error = []
#         for i in range(len(Y_test)):
#             error.append(Y_test[i] - Y_pred[i])
#         squareError = []
#         for i in error:
#             squareError.append(i * i)
#         return sqrt(sum(squareError) / len(squareError))
#     c = mean_absolute_error(Y_test, Y_pred)
#     d = mean_squared_error(Y_test, Y_pred)
#     e = rmse(Y_test, Y_pred)
#     f = r2_score(Y_test, Y_pred)
#     # MAPE
#     for i in range(len(Y_test)):
#         if Y_test[i] == 0:
#             Y_test[i] = Y_test_mean
#     for i in range(len(Y_pred)):
#         if Y_pred[i] == 0:
#             Y_pred[i] = Y_pred_mean
#     def mape(Y_test, Y_pred):
#         n = len(Y_test)
#         mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
#         return mape
#     g = mape(Y_test, Y_pred)
#     MAE.append(c)
#     MSE.append(d)
#     RMSE.append(e)
#     R2.append(f)
#     MAPE.append(g)
#     # 绘图
#     fig = plt.figure()
#     plt.scatter(Y_test, Y_pred)
#     plt.xlabel("Y_test")
#     plt.ylabel("Y_pred")
#     plt.xlim(0, 20)
#     plt.ylim(0, 20)
#     # #plt.show()
# print("SVC:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
#       np.mean(MAPE))


#2.3 NuSVC-支持向量回归
# MAE=[]
# MSE=[]
# RMSE=[]
# R2=[]
# MAPE=[]
# for i in range(10):
#     nusvc = NuSVC(nu=0.5, kernel='rbf', degree=3,
#                   gamma='auto', coef0=0.0, shrinking=True, tol=0.001,
#                   cache_size=200, verbose=False, max_iter=-1, probability=False,
#                   decision_function_shape=None, random_state=None)
#     nusvc.fit(X_train,Y_train)
#     Y_pred=nusvc.predict(X_test)
#     # 计算得分
#     a=nusvc.score(X_train,Y_train)
#     b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
#     Y_train = list(Y_train)
#     Y_test = list(Y_test)
#     Y_pred = list(Y_pred)
#     Y_test = np.array(Y_test)
#     Y_pred = np.array(Y_pred)
#     Y_test_mean = np.mean(Y_test)
#     Y_pred_mean = np.mean(Y_pred)
#
#     # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
#     # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
#     # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
#     def rmse(Y_test, Y_pred):  # RMSE
#         error = []
#         for i in range(len(Y_test)):
#             error.append(Y_test[i] - Y_pred[i])
#         squareError = []
#         for i in error:
#             squareError.append(i * i)
#         return sqrt(sum(squareError) / len(squareError))
#     c = mean_absolute_error(Y_test, Y_pred)
#     d = mean_squared_error(Y_test, Y_pred)
#     e = rmse(Y_test, Y_pred)
#     f = r2_score(Y_test, Y_pred)
#     # MAPE
#     for i in range(len(Y_test)):
#         if Y_test[i] == 0:
#             Y_test[i] = Y_test_mean
#     for i in range(len(Y_pred)):
#         if Y_pred[i] == 0:
#             Y_pred[i] = Y_pred_mean
#     def mape(Y_test, Y_pred):
#         n = len(Y_test)
#         mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
#         return mape
#     g = mape(Y_test, Y_pred)
#     MAE.append(c)
#     MSE.append(d)
#     RMSE.append(e)
#     R2.append(f)
#     MAPE.append(g)
#     # 绘图
#     fig = plt.figure()
#     plt.scatter(Y_test, Y_pred)
#     plt.xlabel("Y_test")
#     plt.ylabel("Y_pred")
#     plt.xlim(0, 20)
#     plt.ylim(0, 20)
#     # #plt.show()
# print("NuSVC:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
#       np.mean(MAPE))


#2.4 LinearSVR-线性支持向量回归
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
for i in range(10):
    linearsvr = LinearSVR(epsilon=0.0, tol=0.0001, C=1.0,
                          loss='epsilon_insensitive',
                          fit_intercept=True, intercept_scaling=1.0,
                          dual=True, verbose=0, random_state=None,
                          max_iter=1000)
    linearsvr.fit(X_train,Y_train)
    Y_pred=linearsvr.predict(X_test)
    # 计算得分
    a=linearsvr.score(X_train,Y_train)
    b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train = list(Y_train)
    Y_test = list(Y_test)
    Y_pred = list(Y_pred)
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)
    Y_test_mean = np.mean(Y_test)
    Y_pred_mean = np.mean(Y_pred)

    # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
    def rmse(Y_test, Y_pred):  # RMSE
        error = []
        for i in range(len(Y_test)):
            error.append(Y_test[i] - Y_pred[i])
        squareError = []
        for i in error:
            squareError.append(i * i)
        return sqrt(sum(squareError) / len(squareError))
    c = mean_absolute_error(Y_test, Y_pred)
    d = mean_squared_error(Y_test, Y_pred)
    e = rmse(Y_test, Y_pred)
    f = r2_score(Y_test, Y_pred)
    # MAPE
    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            Y_test[i] = Y_test_mean
    for i in range(len(Y_pred)):
        if Y_pred[i] == 0:
            Y_pred[i] = Y_pred_mean
    def mape(Y_test, Y_pred):
        n = len(Y_test)
        mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
        return mape
    g = mape(Y_test, Y_pred)
    MAE.append(c)
    MSE.append(d)
    RMSE.append(e)
    R2.append(f)
    MAPE.append(g)
    # 绘图
    fig = plt.figure()
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    # #plt.show()
print("LinearSVR:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
      np.mean(MAPE))



#2.5 SVR-支持向量回归
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
for i in range(10):
    svr = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001,
              C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False,
              max_iter=-1)
    svr.fit(X_train, Y_train)
    Y_pred = svr.predict(X_test)
    # 计算得分
    a=svr.score(X_train,Y_train)
    b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train = list(Y_train)
    Y_test = list(Y_test)
    Y_pred = list(Y_pred)
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)
    Y_test_mean = np.mean(Y_test)
    Y_pred_mean = np.mean(Y_pred)

    # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
    # #R^2->1  模型越好， R^2->0 模型越坏
    def rmse(Y_test, Y_pred):  # RMSE
        error = []
        for i in range(len(Y_test)):
            error.append(Y_test[i] - Y_pred[i])
        squareError = []
        for i in error:
            squareError.append(i * i)
        return sqrt(sum(squareError) / len(squareError))
    c = mean_absolute_error(Y_test, Y_pred)
    d = mean_squared_error(Y_test, Y_pred)
    e = rmse(Y_test, Y_pred)
    f = r2_score(Y_test, Y_pred)
    # MAPE
    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            Y_test[i] = Y_test_mean
    for i in range(len(Y_pred)):
        if Y_pred[i] == 0:
            Y_pred[i] = Y_pred_mean
    def mape(Y_test, Y_pred):
        n = len(Y_test)
        mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
        return mape
    g = mape(Y_test, Y_pred)
    MAE.append(c)
    MSE.append(d)
    RMSE.append(e)
    R2.append(f)
    MAPE.append(g)
    # 绘图
    fig = plt.figure()
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    # #plt.show()
print("SVR:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
      np.mean(MAPE))


#2.6 NuSVR-支持向量回归
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
for i in range(10):
    nusvr = NuSVR(nu=0.5, C=1.0, kernel='rbf', degree=3,
                  gamma='auto', coef0=0.0, shrinking=True, tol=0.001,
                  cache_size=200, verbose=False, max_iter=-1)
    nusvr.fit(X_train,Y_train)
    Y_pred=nusvr.predict(X_test)
    # 计算得分
    a=nusvr.score(X_train,Y_train)
    b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train = list(Y_train)
    Y_test = list(Y_test)
    Y_pred = list(Y_pred)
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)
    Y_test_mean = np.mean(Y_test)
    Y_pred_mean = np.mean(Y_pred)

    # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
    def rmse(Y_test, Y_pred):  # RMSE
        error = []
        for i in range(len(Y_test)):
            error.append(Y_test[i] - Y_pred[i])
        squareError = []
        for i in error:
            squareError.append(i * i)
        return sqrt(sum(squareError) / len(squareError))
    c = mean_absolute_error(Y_test, Y_pred)
    d = mean_squared_error(Y_test, Y_pred)
    e = rmse(Y_test, Y_pred)
    f = r2_score(Y_test, Y_pred)
    # MAPE
    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            Y_test[i] = Y_test_mean
    for i in range(len(Y_pred)):
        if Y_pred[i] == 0:
            Y_pred[i] = Y_pred_mean
    def mape(Y_test, Y_pred):
        n = len(Y_test)
        mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
        return mape
    g = mape(Y_test, Y_pred)
    MAE.append(c)
    MSE.append(d)
    RMSE.append(e)
    R2.append(f)
    MAPE.append(g)
    # 绘图
    fig = plt.figure()
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    # #plt.show()
print("NuSVR:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
      np.mean(MAPE))


#2.7 OneClassSVM
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
for i in range(10):
    oneclasssvm = OneClassSVM(kernel='rbf', degree=3, gamma='auto',
                              coef0=0.0, tol=0.001, nu=0.5, shrinking=True,
                              cache_size=200, verbose=False, max_iter=-1, random_state=None)
    oneclasssvm.fit(X_train,Y_train)
    Y_pred=oneclasssvm.predict(X_test)
    # 计算得分
    # a=oneclasssvm.score(X_train,Y_train)
    b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train = list(Y_train)
    Y_test = list(Y_test)
    Y_pred = list(Y_pred)
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)
    Y_test_mean = np.mean(Y_test)
    Y_pred_mean = np.mean(Y_pred)

    # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
    def rmse(Y_test, Y_pred):  # RMSE
        error = []
        for i in range(len(Y_test)):
            error.append(Y_test[i] - Y_pred[i])
        squareError = []
        for i in error:
            squareError.append(i * i)
        return sqrt(sum(squareError) / len(squareError))
    c = mean_absolute_error(Y_test, Y_pred)
    d = mean_squared_error(Y_test, Y_pred)
    e = rmse(Y_test, Y_pred)
    f = r2_score(Y_test, Y_pred)
    # MAPE
    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            Y_test[i] = Y_test_mean
    for i in range(len(Y_pred)):
        if Y_pred[i] == 0:
            Y_pred[i] = Y_pred_mean
    def mape(Y_test, Y_pred):
        n = len(Y_test)
        mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
        return mape
    g = mape(Y_test, Y_pred)
    MAE.append(c)
    MSE.append(d)
    RMSE.append(e)
    R2.append(f)
    MAPE.append(g)
    # 绘图
    fig = plt.figure()
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    # #plt.show()
print("OneclassSVM:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
      np.mean(MAPE))


'''
####################################################################################Gaussian Naive Bayes
##三、贝叶斯模型
#3.1 GaussianNB-高斯贝叶斯分类器
gaussian=GaussianNB()
gaussian.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=gaussian.predict(X_test.astype('int'))
a=gaussian.score(X_train.astype('int'),Y_train.astype('int'))
b=gaussian.predict_proba(X_train)
c=gaussian.predict_log_proba(X_train)
print("Gaussian Naive Bayes:",a,b,c)

#3.2 MultinomialNB-多项式贝叶斯分类器
multinomialnb=MultinomialNB(alpha=1.0,fit_prior=True,class_prior=None)
multinomialnb.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=multinomialnb.predict(X_test.astype('int'))
a=multinomialnb.score(X_train.astype('int'),Y_train.astype('int'))
b=multinomialnb.predict_proba(X_train)
c=multinomialnb.predict_log_proba(X_train)
print("MultinomialNB:",a,b,c)

#3.3 BernoulliNB-伯努利贝叶斯分类器
bernoullinb=BernoulliNB(alpha=1.0,binarize=0.0,fit_prior=True,
                        class_prior=None)
bernoullinb.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=bernoullinb.predict(X_test.astype('int'))
a=bernoullinb.score(X_train.astype('int'),Y_train.astype('int'))
b=bernoullinb.predict_proba(X_train)
c=bernoullinb.predict_log_proba(X_train)
print("BernoulliNB:",a,b,c)
'''



##################################################################################Random Forest
#四、决策树
# 4.1 DecisionTreeRegressor-回归决策树
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
for i in range(10):
    decisiontree = DecisionTreeRegressor(criterion='mse', splitter='best',
                                         max_depth=None, min_impurity_split=2,
                                         min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                         max_features=None, random_state=None,
                                         max_leaf_nodes=None, presort=False)
    decisiontree.fit(X_train,Y_train)
    Y_pred=decisiontree.predict(X_test)
    # 计算得分
    a=decisiontree.score(X_train,Y_train)
    b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train = list(Y_train)
    Y_test = list(Y_test)
    Y_pred = list(Y_pred)
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)
    Y_test_mean = np.mean(Y_test)
    Y_pred_mean = np.mean(Y_pred)

    # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
    def rmse(Y_test, Y_pred):  # RMSE
        error = []
        for i in range(len(Y_test)):
            error.append(Y_test[i] - Y_pred[i])
        squareError = []
        for i in error:
            squareError.append(i * i)
        return sqrt(sum(squareError) / len(squareError))
    c = mean_absolute_error(Y_test, Y_pred)
    d = mean_squared_error(Y_test, Y_pred)
    e = rmse(Y_test, Y_pred)
    f = r2_score(Y_test, Y_pred)
    # MAPE
    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            Y_test[i] = Y_test_mean
    for i in range(len(Y_pred)):
        if Y_pred[i] == 0:
            Y_pred[i] = Y_pred_mean
    def mape(Y_test, Y_pred):
        n = len(Y_test)
        mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
        return mape
    g = mape(Y_test, Y_pred)
    MAE.append(c)
    MSE.append(d)
    RMSE.append(e)
    R2.append(f)
    MAPE.append(g)
    # 绘图
    fig = plt.figure()
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    # #plt.show()
print("DecisionTree:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
      np.mean(MAPE))


# #4.2 DecisionTreeClassifier-分类决策树
# decisiontree_classifier=DecisionTreeClassifier(criterion='gini',
#                                                splitter='best',
#                                                max_depth=None,
#                                                min_impurity_split=2,
#                                                min_samples_leaf=1,
#                                                min_weight_fraction_leaf=0.0,
#                                                max_features=None,random_state=None,
#                                                max_leaf_nodes=None,class_weight=None,presort=False)
# decisiontree_classifier.fit(X_train.astype('int'),Y_train.astype('int'))
# Y_pred=decisiontree_classifier.predict(X_test.astype('int'))
# a=decisiontree_classifier.score(X_train.astype('int'),Y_train.astype('int'))
# b=decisiontree_classifier.predict_proba(X_train)
# c=decisiontree_classifier.predict_log_proba(X_train)
# d=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
# print("BernoulliNB:",a,b,c,d)
'''

'''
####################################################################################KNN
#五、KNN
#5.1 KNeighborsClassifier-knn分类模型
# knn=KNeighborsClassifier(n_neighbors=5,weights='uniform',
#                          algorithm='auto',leaf_size=30,
#                          p=2,metric='minkowski',metric_params=None,
#                          n_jobs=1)
# knn.fit(X_train.astype('int'),Y_train.astype('int'))
# Y_pred=knn.predict(X_test.astype('int'))
# a=knn.score(X_train.astype('int'),Y_train.astype('int'))
# b=knn.predict_proba(X_train)
# c=knn.kneighbors()
# d=knn.kneighbors_graph()
# print("KNN:",a,b)

#5.2 KNeighborsRegressor-knn回归模型
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
for i in range(10):
    knn_regressor = KNeighborsRegressor(n_neighbors=5, weights='uniform',
                                        algorithm='auto', leaf_size=30, p=2,
                                        metric_params=None, metric='minkowski', n_jobs=1)
    knn_regressor.fit(X_train,Y_train)
    Y_pred=knn_regressor.predict(X_test)
    # 计算得分
    a=knn_regressor.score(X_train,Y_train)
    b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train = list(Y_train)
    Y_test = list(Y_test)
    Y_pred = list(Y_pred)
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)
    Y_test_mean = np.mean(Y_test)
    Y_pred_mean = np.mean(Y_pred)

    # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
    def rmse(Y_test, Y_pred):  # RMSE
        error = []
        for i in range(len(Y_test)):
            error.append(Y_test[i] - Y_pred[i])
        squareError = []
        for i in error:
            squareError.append(i * i)
        return sqrt(sum(squareError) / len(squareError))
    c = mean_absolute_error(Y_test, Y_pred)
    d = mean_squared_error(Y_test, Y_pred)
    e = rmse(Y_test, Y_pred)
    f = r2_score(Y_test, Y_pred)
    # MAPE
    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            Y_test[i] = Y_test_mean
    for i in range(len(Y_pred)):
        if Y_pred[i] == 0:
            Y_pred[i] = Y_pred_mean
    def mape(Y_test, Y_pred):
        n = len(Y_test)
        mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
        return mape
    g = mape(Y_test, Y_pred)
    MAE.append(c)
    MSE.append(d)
    RMSE.append(e)
    R2.append(f)
    MAPE.append(g)
    # 绘图
    fig = plt.figure()
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    # #plt.show()
print("KNN_Regressor:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
      np.mean(MAPE))


##################################################################################################AdaBoost
#六、AdaBoost
#6.1 AdaBoostClassifier-分类器
# AdaBoostclassifier=AdaBoostClassifier(base_estimator=None,n_estimators=50,
#                             learning_rate=1.0,algorithm='SAMME.R',
#                             random_state=None)
# AdaBoostclassifier.fit(X_train.astype('int'),Y_train.astype('int'))
# Y_pred=AdaBoostclassifier.predict(X_test.astype('int'))
# a=AdaBoostclassifier.score(X_train.astype('int'),Y_train.astype('int'))
# b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
# # b=AdaBoostclassifier.predict_proba(X_train)
# # c=AdaBoostclassifier.predict_log_proba(X_train)
# # d=AdaBoostclassifier.staged_predict(X_train)
# # e=AdaBoostclassifier.staged_predict_proba(X_train)
# # f=AdaBoostclassifier.staged_score(X_train,Y_train)
# print("Adaboost Classifier:",a,b)


#6.2 AdaBoostRegressor-回归器
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
for i in range(10):
    AdaBoostregressor = AdaBoostRegressor(base_estimator=None, n_estimators=20,
                                          learning_rate=0.5, loss='linear', random_state=None)
    AdaBoostregressor.fit(X_train,Y_train)
    Y_pred=AdaBoostregressor.predict(X_test)
    # 计算得分
    a=AdaBoostregressor.score(X_train,Y_train)
    b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train = list(Y_train)
    Y_test = list(Y_test)
    Y_pred = list(Y_pred)
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)
    Y_test_mean = np.mean(Y_test)
    Y_pred_mean = np.mean(Y_pred)

    # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
    def rmse(Y_test, Y_pred):  # RMSE
        error = []
        for i in range(len(Y_test)):
            error.append(Y_test[i] - Y_pred[i])
        squareError = []
        for i in error:
            squareError.append(i * i)
        return sqrt(sum(squareError) / len(squareError))
    c = mean_absolute_error(Y_test, Y_pred)
    d = mean_squared_error(Y_test, Y_pred)
    e = rmse(Y_test, Y_pred)
    f = r2_score(Y_test, Y_pred)
    # MAPE
    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            Y_test[i] = Y_test_mean
    for i in range(len(Y_pred)):
        if Y_pred[i] == 0:
            Y_pred[i] = Y_pred_mean
    def mape(Y_test, Y_pred):
        n = len(Y_test)
        mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
        return mape
    g = mape(Y_test, Y_pred)
    MAE.append(c)
    MSE.append(d)
    RMSE.append(e)
    R2.append(f)
    MAPE.append(g)
    # 绘图
    fig = plt.figure()
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    # #plt.show()
print("AdaBoostRegressor:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
      np.mean(MAPE))


##############################################################
#七、梯度提升树
#7.1 GradientBoostingClassifier-GBDT分类模型
# MAE=[]
# MSE=[]
# RMSE=[]
# R2=[]
# MAPE=[]
# for i in range(10):
#     gbdt=GradientBoostingClassifier(loss='deviance',learning_rate=0.1,
#                                     n_estimators=100,subsample=1.0,
#                                     min_samples_split=2,min_samples_leaf=1,
#                                     min_weight_fraction_leaf=0.0,max_depth=3,init=None,
#                                     random_state=None,max_features=None,verbose=0,max_leaf_nodes=None,
#                                     warm_start=False,presort='auto')
#     gbdt.fit(X_train,Y_train)
#     Y_pred=gbdt.predict(X_test)
# #计算得分
#     # a=random_forest_Regressor.score(X_train,Y_train)
#     # b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
#     Y_train=list(Y_train)
#     Y_test=list(Y_test)
#     Y_pred=list(Y_pred)
#     Y_test=np.array(Y_test)
#     Y_pred=np.array(Y_pred)
#     Y_test_mean=np.mean(Y_test)
#     Y_pred_mean=np.mean(Y_pred)
#     # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
#     # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
#     # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
#     def rmse(Y_test,Y_pred):#RMSE
#         error=[]
#         for i in range(len(Y_test)):
#             error.append(Y_test[i]-Y_pred[i])
#         squareError=[]
#         for i in error:
#             squareError.append(i*i)
#         return sqrt(sum(squareError)/len(squareError))
#
#     c = mean_absolute_error(Y_test, Y_pred)
#     d = mean_squared_error(Y_test, Y_pred)
#     e = rmse(Y_test, Y_pred)
#     f = r2_score(Y_test, Y_pred)
#     #MAPE
#     for i in range(len(Y_test)):
#         if Y_test[i]==0:
#             Y_test[i]=Y_test_mean
#     for i in range(len(Y_pred)):
#         if Y_pred[i]==0:
#             Y_pred[i]=Y_pred_mean
#     def mape(Y_test,Y_pred):
#         n=len(Y_test)
#         mape=sum(np.abs((Y_test-Y_pred)/Y_test))/n*100
#         return mape
#
#     g=mape(Y_test,Y_pred)
#     MAE.append(c)
#     MSE.append(d)
#     RMSE.append(e)
#     R2.append(f)
#     MAPE.append(g)
#     # 绘图
#     fig = plt.figure()
#     plt.scatter(Y_test,Y_pred)
#     plt.xlabel("Y_test")
#     plt.ylabel("Y_pred")
#     plt.xlim(0, 20)
#     plt.ylim(0, 20)
#     #plt.show()
# print("GBDTClassifier:","MAE:",np.mean(MAE),"MSE:",np.mean(MSE),"RMSE:",np.mean(RMSE),"R2_Score:",np.mean(R2),"MAPE:",np.mean(MAPE))



#7.2 GradientBoostingRegressor-GBDT回归模型
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
for i in range(10):
    gbdtregressor=GradientBoostingRegressor(loss='ls',learning_rate=0.1,
                                    n_estimators=100,subsample=1.0,
                                    min_samples_split=2,min_samples_leaf=1,
                                    min_weight_fraction_leaf=0.0,max_depth=3,init=None,
                                    random_state=None,max_features=None,verbose=0,max_leaf_nodes=None,
                                    warm_start=False,presort='auto')
    gbdtregressor.fit(X_train,Y_train)
    Y_pred=gbdtregressor.predict(X_test)
    # 计算得分
    # a=random_forest_Regressor.score(X_train,Y_train)
    # b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train = list(Y_train)
    Y_test = list(Y_test)
    Y_pred = list(Y_pred)
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)
    Y_test_mean = np.mean(Y_test)
    Y_pred_mean = np.mean(Y_pred)


    # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
    def rmse(Y_test, Y_pred):  # RMSE
        error = []
        for i in range(len(Y_test)):
            error.append(Y_test[i] - Y_pred[i])
        squareError = []
        for i in error:
            squareError.append(i * i)
        return sqrt(sum(squareError) / len(squareError))

    c = mean_absolute_error(Y_test, Y_pred)
    d = mean_squared_error(Y_test, Y_pred)
    e = rmse(Y_test, Y_pred)
    f = r2_score(Y_test, Y_pred)
    # MAPE
    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            Y_test[i] = Y_test_mean
    for i in range(len(Y_pred)):
        if Y_pred[i] == 0:
            Y_pred[i] = Y_pred_mean


    def mape(Y_test, Y_pred):
        n = len(Y_test)
        mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / n * 100
        return mape


    g = mape(Y_test, Y_pred)
    MAE.append(c)
    MSE.append(d)
    RMSE.append(e)
    R2.append(f)
    MAPE.append(g)
    # 绘图
    fig = plt.figure()
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    #plt.show()
print("GBDTRegrssor:", "MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:",
      np.mean(R2), "MAPE:", np.mean(MAPE))


########################################################################
#八、Random Forest
#8.1 RandomForestClassifier-随机森林分类模型
# random_forest_classifier=RandomForestClassifier(n_estimators=25,criterion='entropy',
#                                      max_depth=None,min_samples_split=2,
#                                      min_samples_leaf=1,min_weight_fraction_leaf=0.0,
#                                      max_features='auto',max_leaf_nodes=None,bootstrap=True,
#                                      oob_score=True,n_jobs=1,random_state=None,verbose=0,
#                                      warm_start=True,class_weight=None)
# random_forest_classifier.fit(X_train.astype('int'),Y_train.astype('int'))
# Y_pred=random_forest_classifier.predict(X_test.astype('int'))
#
# a=random_forest_classifier.score(X_train.astype('int'),Y_train.astype('int'))
# b=random_forest_classifier.predict_log_proba(X)
# c=random_forest_classifier.predict_proba(X)
# d=random_forest_classifier.feature_importances_
# e=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
# f=classification_report(Y_test.astype('int'),Y_pred.astype('int'))
# g=confusion_matrix(Y_test.astype('int'),Y_pred.astype('int'))
# h=mean_absolute_error(Y_test.astype('int'),Y_pred.astype('int'))
# i=mean_squared_error(Y_test.astype('int'),Y_pred.astype('int'))
# j=r2_score(Y_test.astype('int'),Y_pred.astype('int'))
# k=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'),normalize=False)
#
# Machine_Learning_Results["Random_forest_classifier_score:"]=a
# # Machine_Learning_Results["Random_forest_classifier_predict_log_proba:"]=b
# # Machine_Learning_Results["Random_forest_classifier_predict_proba:"]=c
# Machine_Learning_Results["Random_forest_classifier_feature_importances_:"]=d
# Machine_Learning_Results["Random_forest_classifier_accuracy_score:"]=e
# # Machine_Learning_Results["Random_forest_classifier_classification_report:"]=f
# # Machine_Learning_Results["Random_forest_classifier_confusion_matrix:"]=g
# # Machine_Learning_Results["Random_forest_classifier_mae:"]=h
# # Machine_Learning_Results["Random_forest_classifier_mse:"]=i
# # Machine_Learning_Results["Random_forest_classifier_r2_score:"]=j
# Machine_Learning_Results["Random_forest_classifier_accuracy_score_sample:"]=k
# print(Machine_Learning_Results)


#特征重要性
# color=sns.color_palette()
# sns.set_style('darkgrid')
# feature_list=X_train
# feature_importance=random_forest_classifier.feature_importances_
# sorted_idx=np.argsort(feature_importance)
# plt.figure(figsize=(5,7))
# plt.barh(range(len(sorted_idx)),feature_importance[sorted_idx],
#          align='center')
# plt.yticks(range(len(sorted_idx)),feature_list[sorted_idx])
# plt.xlabel('Importance')
# plt.title('Feature Importance')
# plt.draw()
# #plt.show()



#8.2 RandomForestRegressor-随机森林回归模型

#Random Forest Regression
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
for i in range(10):
    random_forest_Regressor=RandomForestRegressor(n_estimators=20,criterion='mse',
                                         max_depth=None,min_samples_split=2,
                                         min_samples_leaf=1,min_weight_fraction_leaf=0.0,
                                         max_features='auto',max_leaf_nodes=None,bootstrap=True,
                                         oob_score=False,n_jobs=1,random_state=None,verbose=0,warm_start=False)
    random_forest_Regressor.fit(X_train,Y_train)
    Y_pred=random_forest_Regressor.predict(X_test)

    #计算得分
    # a=random_forest_Regressor.score(X_train,Y_train)
    # b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train=list(Y_train)
    Y_test=list(Y_test)
    Y_pred=list(Y_pred)
    Y_test=np.array(Y_test)
    Y_pred=np.array(Y_pred)
    Y_test_mean=np.mean(Y_test)
    Y_pred_mean=np.mean(Y_pred)
    # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
    def rmse(Y_test,Y_pred):#RMSE
        error=[]
        for i in range(len(Y_test)):
            error.append(Y_test[i]-Y_pred[i])
        squareError=[]
        for i in error:
            squareError.append(i*i)
        return sqrt(sum(squareError)/len(squareError))

    c = mean_absolute_error(Y_test, Y_pred)
    d = mean_squared_error(Y_test, Y_pred)
    e = rmse(Y_test, Y_pred)
    f = r2_score(Y_test, Y_pred)
    #MAPE
    for i in range(len(Y_test)):
        if Y_test[i]==0:
            Y_test[i]=Y_test_mean
    for i in range(len(Y_pred)):
        if Y_pred[i]==0:
            Y_pred[i]=Y_pred_mean
    def mape(Y_test,Y_pred):
        n=len(Y_test)
        mape=sum(np.abs((Y_test-Y_pred)/Y_test))/n*100
        return mape

    g=mape(Y_test,Y_pred)
    MAE.append(c)
    MSE.append(d)
    RMSE.append(e)
    R2.append(f)
    MAPE.append(g)
    # 绘图
    fig = plt.figure()
    plt.scatter(Y_test,Y_pred)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    # #plt.show()
print("RandomForestRegression:","MAE:",np.mean(MAE),"MSE:",np.mean(MSE),"RMSE:",np.mean(RMSE),"R2_Score:",np.mean(R2),"MAPE:",np.mean(MAPE))


#九、XGBoost
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
for i in range(10):
    xgboost=XGBRegressor(max_depth=1,learning_rate=0.1,n_estimators=100,
                         silent=True,objective='reg:linear',booster='gbtree',n_jobs=1,
                         gamma=0,min_child_weight=1,max_delta_step=0,subsample=1,colsample_bytree=1,
                         colsample_bylevel=1,reg_alpha=0,reg_lambda=1,scale_pos_weight=1,
                         base_score=0.5,random_state=0,missing=None)
    xgboost.fit(X_train,Y_train)
    Y_pred=xgboost.predict(X_test)

    #计算得分
    # a=random_forest_Regressor.score(X_train,Y_train)
    # b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train=list(Y_train)
    Y_test=list(Y_test)
    Y_pred=list(Y_pred)
    Y_test=np.array(Y_test)
    Y_pred=np.array(Y_pred)
    Y_test_mean=np.mean(Y_test)
    Y_pred_mean=np.mean(Y_pred)
    # #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    # #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    # #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
    def rmse(Y_test,Y_pred):#RMSE
        error=[]
        for i in range(len(Y_test)):
            error.append(Y_test[i]-Y_pred[i])
        squareError=[]
        for i in error:
            squareError.append(i*i)
        return sqrt(sum(squareError)/len(squareError))

    c = mean_absolute_error(Y_test, Y_pred)
    d = mean_squared_error(Y_test, Y_pred)
    e = rmse(Y_test, Y_pred)
    f = r2_score(Y_test, Y_pred)
    #MAPE
    for i in range(len(Y_test)):
        if Y_test[i]==0:
            Y_test[i]=Y_test_mean
    for i in range(len(Y_pred)):
        if Y_pred[i]==0:
            Y_pred[i]=Y_pred_mean
    def mape(Y_test,Y_pred):
        n=len(Y_test)
        mape=sum(np.abs((Y_test-Y_pred)/Y_test))/n*100
        return mape

    g=mape(Y_test,Y_pred)
    MAE.append(c)
    MSE.append(d)
    RMSE.append(e)
    R2.append(f)
    MAPE.append(g)
    # 绘图
    fig = plt.figure()
    plt.scatter(Y_test,Y_pred)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred")
    plt.xlim(0,20)
    plt.ylim(0,20)
    # #plt.show()
print("XGBoost:","MAE:",np.mean(MAE),"MSE:",np.mean(MSE),"RMSE:",np.mean(RMSE),"R2_Score:",np.mean(R2),"MAPE:",np.mean(MAPE))

# with open('1.txt','w') as f:
#     json.dump(Machine_Learning_Results,f)
# #特征重要性
#     color=sns.color_palette()
#     sns.set_style('darkgrid')
#     feature_list=X_train
#     feature_importance=svr.feature_importances_
#     sorted_idx=np.argsort(feature_importance)
#     plt.figure(figsize=(5,7))
#     plt.barh(range(len(sorted_idx)),feature_importance[sorted_idx],align='center')
#     plt.yticks(range(len(sorted_idx)),feature_list[sorted_idx])
#     plt.xlabel('Importance')
#     plt.title('Feature Importance')
#     plt.draw()
#     plt.show()