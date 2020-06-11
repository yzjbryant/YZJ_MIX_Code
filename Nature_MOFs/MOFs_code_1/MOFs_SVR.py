import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,cross_val_score,cross_validate
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
# import xgboost as xgb
# from xgboost import XGBRegressor
from math import sqrt
#分类指标
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#回归指标
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,normalize

data=pd.read_excel('CIF_Parameters.xlsx',header=0)

Target=['Energy_C4H4S_298K_10kpa','Energy_C4H4S_363K_10kpa','Energy_C4H4S_363K_100kpa',
        'Energy_C6H6_298K_10kpa','Energy_C6H6_363K_10kpa','Energy_C6H6_363K_100kpa',
        'Loading_C4H4S_298K_10kpa','Loading_C4H4S_363K_10kpa','Loading_C4H4S_363K_100kpa',
        'Loading_C6H6_298K_10kpa','Loading_C6H6_363K_10kpa','Loading_C6H6_363K_100kpa']#Number of Target(i) is 12
Feature_Descriptor=['Materials','Cluster','Linker','Group','Length_a','Length_b','Length_c','Angle_alpha','Angle_beta',
                    'Angle_gamma','Number of Element','Tong','Zn','C','H','Br','Lu','I','N','O','F','S','Di','Df','Dif',
                    'Unitcell_volume','Density','ASA_A^2','ASA_m^2/cm^3','ASA_m^2/g','NASA_A^2','NASA_m^2/cm^3',
                    'NASA_m^2/g','AV_A^3','AV_Volume_fraction','AV_cm^3/g','NAV_A^3','NAV_Volume_fraction',
                    'NAV_cm^3/g','Energy_C4H4S_298K_10kpa','Energy_C4H4S_363K_10kpa','Energy_C4H4S_363K_100kpa',
                    'Energy_C6H6_298K_10kpa','Energy_C6H6_363K_10kpa','Energy_C6H6_363K_100kpa','Loading_C4H4S_298K_10kpa',
                    'Loading_C4H4S_363K_10kpa','Loading_C4H4S_363K_100kpa','Loading_C6H6_298K_10kpa','Loading_C6H6_363K_10kpa',
                    'Loading_C6H6_363K_100kpa']

X=data.drop(['Materials','Energy_C4H4S_298K_10kpa','Energy_C4H4S_363K_10kpa','Energy_C4H4S_363K_100kpa',
                    'Energy_C6H6_298K_10kpa','Energy_C6H6_363K_10kpa','Energy_C6H6_363K_100kpa','Loading_C4H4S_298K_10kpa',
                    'Loading_C4H4S_363K_10kpa','Loading_C4H4S_363K_100kpa','Loading_C6H6_298K_10kpa','Loading_C6H6_363K_10kpa',
                    'Loading_C6H6_363K_100kpa'],axis=1)
#Characteristic数据标准化
scaler=StandardScaler(copy=True,with_mean=True,with_std=True).fit(X)
X=scaler.transform(X)

for j in range(len(Target)):
#     '''
#     #数据可视化
#     correlations=X.corr()
#     correction=abs(correlations)
#     fig=plt.figure()
#     ax=fig.add_subplot(figsize=(20,20))
#     ax1=sns.heatmap(correction) #相关性热力图
#     plt.show()
#     '''
    Y=data[Target[j]]
    #Energy、Loading数据采用正则化
    normalizer=preprocessing.Normalizer(copy=True,norm='l2').fit(Y.values.reshape(1,-1))
    Y=normalizer.transform(Y.values.reshape(1,-1))
    Y=np.transpose(Y)
    # print(X.shape,Y.shape)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=None)
    print(X_train.shape,Y_train.shape)
    print(X_test,Y_test)
    # print(X_train,"\n++++++++++",X_test,"\n+++++++++++",Y_train,"\n++++++++++++",Y_test)
    MAE=[]
    MSE=[]
    RMSE=[]
    R2=[]
    MAPE=[]
    svr = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001,
              C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False,
              max_iter=-1)
    svr.fit(X_train, Y_train)
    Y_pred = svr.predict(X_test)
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
    print(Target[j],"SVR:","MAE:", np.mean(MAE), "MSE:", np.mean(MSE), "RMSE:", np.mean(RMSE), "R2_Score:", np.mean(R2), "MAPE:",
    np.mean(MAPE))
    kf=KFold(n_splits=10)
    for train,test in kf.split(X):
        print("K折划分：%s %s"%(train.shape,test.shape))
        break
    # scores=cross_val_score(svr,X_test.astype('float64'),Y_pred.astype('float64'),cv=10,scoring='accuracy')
    # print(scores)
    '''
    with open('result.txt','w') as f:
        for i in range(len(Target)):
            f.write(Target[i])
            f.write('\n')
            f.write("MAE: ")
            f.write(str(np.mean(MAE)))
            f.write('\t')
            f.write("MSE: ")
            f.write(str(np.mean(MSE)))
            f.write('\t')
            f.write("RMSE: ")
            f.write(str(np.mean(RMSE)))
            f.write('\t')
            f.write("R2_Score: ")
            f.write(str(np.mean(R2)))
            f.write('\t')
            f.write("MAPE: ")
            f.write(str(np.mean(MAPE)))
            f.write('\n')
    '''
    '''
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
    poly2_reg = PolynomialRegression(degree=20)
    plot_learning_curve(poly2_reg, X_train, X_test, Y_train, Y_test)
    '''