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
#import xgboost as xgb
#from xgboost import XGBRegressor
from math import sqrt
#分类指标
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#回归指标
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

data=pd.read_excel('CIF_Parameters.xlsx',header=0)

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
             'Loading_C4H4S_298K_10kpa','Loading_C4H4S_363K_10kpa','Loading_C4H4S_363K_100kpa',
             'Loading_C6H6_298K_10kpa','Loading_C6H6_363K_10kpa','Loading_C6H6_363K_100kpa'],axis=1)
Y=data['Loading_C4H4S_298K_10kpa']
#数据可视化
correlations=X.corr()
correction=abs(correlations)
fig=plt.figure()
ax=fig.add_subplot(figsize=(20,20))
ax1=sns.heatmap(correction) #相关性热力图
# plt.show()

#定义训练集与测试集
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=None)
# print(X_train,X_test,Y_train,Y_test)
#标准化
# scaler=StandardScaler(copy=True,with_mean=True,with_std=True).fit(X_train)
# X_train=scaler.transform(X_train)
# X_test=scaler.transform(X_test)

#一、线性模型
#1.1 LinearRegression-线性回归模型
MAE=[]
MSE=[]
RMSE=[]
R2=[]
MAPE=[]
import multiprocessing
for i in range(10):
    cores=multiprocessing.cpu_count()
    print(cores)
    pool=multiprocessing.Pool(processes=cores)
    linear=LinearRegression(fit_intercept=True,normalize=True,
                            copy_X=True,n_jobs=1)
    pool.apply_async(linear)
    pool.apply_async(linear.fit(X_train,Y_train))
    Y_pred=linear.predict(X_test)
    pool.apply_async(Y_pred)
    # 计算得分
    # a=linear.score(X_train,Y_train)   #回归不需要得分
    # b=accuracy_score(Y_test.astype('int'),Y_pred.astype('int'))
    Y_train = list(Y_train)
    Y_test = list(Y_test)
    Y_pred = list(Y_pred)
    Y_test = np.array(Y_test)
    Y_pred = np.array(Y_pred)
    Y_test_mean = np.mean(Y_test)
    Y_pred_mean = np.mean(Y_pred)

    #平均绝对误差(MAE)，MAE值越小，说明预测模型拥有更好的精确度
    #均方误差(MSE)，误差的平方的期望值，MSE值越小，预测模型拥有更好的精确度
    #平均绝对百分比误差(MAPE)，偏离，MAPE值越小，说明预测模型拥有更好的精确度
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
    # plt.show()

    def plot_learning_curve(algo, X_train, X_test, y_train, y_test):
        """绘制学习曲线：只需要传入算法(或实例对象)、X_train、X_test、y_train、y_test"""
        """当使用该函数时传入算法，该算法的变量要进行实例化，如：PolynomialRegression(degree=2)，变量 degree 要进行实例化"""
        train_score = []
        test_score = []
        for i in range(1, len(X_train) + 1):
            algo.fit(X_train[:i], y_train[:i])
            y_train_predict = algo.predict(X_train[:i])
            train_score.append(mean_squared_error(y_train[:i], y_train_predict))
            print("train_score:",train_score)
            y_test_predict = algo.predict(X_test)
            test_score.append(mean_squared_error(y_test, y_test_predict))
            print("test_score:", test_score)
        plt.plot([i for i in range(1, len(X_train) + 1)],
                 np.sqrt(train_score), label="train")
        plt.plot([i for i in range(1, len(X_train) + 1)],
                 np.sqrt(test_score), label="test")
        plt.legend()
        plt.axis([0, len(X_train) + 1, 0, 4])
        plt.show()
        plt.savefig('./test1.jpg')
    def PolynomialRegression(degree):
        return Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('std_scaler', StandardScaler()),
            ('lin_reg', LinearRegression())])
print("LinearRegression:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
      np.mean(MAPE))

cores=multiprocessing.cpu_count()
pool=multiprocessing.Pool(processes=cores)
pool.apply_async(PolynomialRegression(degree=20))
pool.apply_async(plot_learning_curve(pool.apply_async(PolynomialRegression(degree=20)), X_train, X_test, Y_train, Y_test))
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