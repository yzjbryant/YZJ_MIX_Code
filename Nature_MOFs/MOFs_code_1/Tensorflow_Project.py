from __future__ import print_function
import sys
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

#导入数据
data=pd.read_excel('CIF_Parameters.xlsx',header=0)
pd.set_option('display.max_rows', 6000)
pd.set_option('display.max_columns', 6000)
pd.set_option('display.width', 1000)

Feature_Descriptor=['Cluster','Linker','Group','Length_a','Length_b','Length_c','Angle_alpha','Angle_beta',
                    'Angle_gamma','Number of Element','Tong','Zn','C','H','Br','Lu','I','N','O','F','S','Di','Df','Dif',
                    'Unitcell_volume','Density','ASA_A^2','ASA_m^2/cm^3','ASA_m^2/g','NASA_A^2','NASA_m^2/cm^3',
                    'NASA_m^2/g','AV_A^3','AV_Volume_fraction','AV_cm^3/g','NAV_A^3','NAV_Volume_fraction',
                    'NAV_cm^3/g','Energy_C4H4S_298K_10kpa','Energy_C4H4S_363K_10kpa','Energy_C4H4S_363K_100kpa',
                    'Energy_C6H6_298K_10kpa','Energy_C6H6_363K_10kpa','Energy_C6H6_363K_100kpa','Loading_C4H4S_298K_10kpa',
                    'Loading_C4H4S_363K_10kpa','Loading_C4H4S_363K_100kpa','Loading_C6H6_298K_10kpa','Loading_C6H6_363K_10kpa',
                    'Loading_C6H6_363K_100kpa']

Target=['Loading_C4H4S_298K_10kpa','Loading_C4H4S_363K_10kpa','Loading_C4H4S_363K_100kpa',
        'Loading_C6H6_298K_10kpa','Loading_C6H6_363K_10kpa','Loading_C6H6_363K_100kpa']
for i in range(len(Feature_Descriptor)):
    print(Feature_Descriptor[i])
    X=data[Feature_Descriptor[i]].values.reshape(-1,1)

# X=data.drop(['Materials','Cluster','Linker','Group','Length_a','Length_b','Length_c','Angle_alpha','Angle_beta',
#                     'Angle_gamma','Number of Element','Zn','C','H','Br','Lu','I','N','O','F','S','Di','Df','Dif',
#                     'Unitcell_volume','Density','ASA_A^2','ASA_m^2/cm^3','ASA_m^2/g','NASA_A^2','NASA_m^2/cm^3',
#                     'NASA_m^2/g','AV_A^3','AV_Volume_fraction','NAV_A^3','NAV_Volume_fraction',
#                     'NAV_cm^3/g','Energy_C4H4S_298K_10kpa','Energy_C4H4S_363K_10kpa','Energy_C4H4S_363K_100kpa',
#                     'Energy_C6H6_298K_10kpa','Energy_C6H6_363K_10kpa','Energy_C6H6_363K_100kpa','Loading_C4H4S_298K_10kpa',
#                     'Loading_C4H4S_363K_10kpa','Loading_C4H4S_363K_100kpa','Loading_C6H6_298K_10kpa','Loading_C6H6_363K_10kpa',
#                     'Loading_C6H6_363K_100kpa','AV_cm^3/g'],axis=1)

    Y=data[Target[0]]
    #定义训练集与测试集
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=None)
    # print(X_train,X_test,Y_train,Y_test)
    print(X_train)
    #数据预处理
    #标准化
    scaler=StandardScaler(copy=True,with_mean=True,with_std=True).fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)

    #训练
    learning_rate=0.1
    training_epochs=50
    display_step=50
    rng=np.random
    n_samples=X_train.shape[0]

    #tf Graph Input
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    X=tf.placeholder("float")
    Y=tf.placeholder("float")

    W=tf.Variable(rng.randn(),name="weight")
    b=tf.Variable(rng.randn(),name="bias")

    pred=tf.add(tf.multiply(X,W),b)

    cost=tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)
    optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            for (x,y) in zip(X_train,Y_train):
                sess.run(optimizer,feed_dict={X:x,Y:y})
            if (epoch+1)%display_step==0:
                c=sess.run(cost,feed_dict={X:X_train,Y:Y_train})
                print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(c),"W=",sess.run(W),"b=",sess.run(b))
        print("Optimization Finished!")
        training_cost=sess.run(cost,feed_dict={X:X_train,Y:Y_train})
        print("Training cost=",training_cost,"W=",sess.run(W),"b=",sess.run(b),'\n')

        plt.plot(X_train,Y_train,'ro',label='Original data')
        plt.plot(X_train,sess.run(W)*X_train+sess.run(b),label='Fitted line')
        plt.xlabel('%s'%Feature_Descriptor[i])
        plt.ylabel('%s'%Target[0])
        plt.legend()
        plt.savefig('F:/MOFS_Extra_Data/Tensorflow_picture/first%s.png'%Feature_Descriptor[i])
        # plt.show()

        print("Testing...(Mean square loss Comparison")
        testing_cost=sess.run(tf.reduce_sum(tf.pow(pred-Y,2))/(2*X_test.shape[0]),
                              feed_dict={X:X_test,Y:Y_test})
        print("Testing cost=",testing_cost)
        print("Absolute mean square loss difference:",abs(training_cost-testing_cost))
        plt.plot(X_test, Y_test, 'bo', label='Testing data')
        plt.plot(X_train, sess.run(W) * X_train + sess.run(b), label='Fitted line')
        plt.xlabel('%s' % Feature_Descriptor[i])
        plt.ylabel('%s' % Target[0])
        plt.legend()
        plt.savefig('F:/MOFS_Extra_Data/Tensorflow_picture/second%s.png' % Feature_Descriptor[i])
        # plt.show()
'''    
#控制台输出文件
# class Logger(object):
#     def __init__(self,fileN="Default.log"):
#         self.terminal=sys.stdout
#         self.log=open(fileN,"a")
#     def write(self,message):
#         self.terminal.write(message)
#         self.log.write(message)
#     def flush(self):
#         pass
# sys.stdout=Logger("./1.txt")
# print("111111")
'''
