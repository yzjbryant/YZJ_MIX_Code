import numpy as np
import pandas as pd
import os
import sys
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
# 回归指标
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,normalize

data=pd.read_excel('CIF_Parameters.xlsx',header=0)
'''
Feature_Descriptor=data.columns
print(Feature_Descriptor)
'''
Target=['Pure_C6H6_Heat_298K_9.99kpa',
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
       'Working_capacity_TSA_10kpa_298K_363K']#Number of Target(i) is 29
Feature_Descriptor=['Materials','Cluster', 'Linker', 'Group', 'Mass', 'Length_a',
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
       'Working_capacity_TSA_10kpa_298K_363K'] #Number of Feature_Descriptor is 69
data=data.drop(['Materials'],axis=1)



# correlations=data.corr()
# correction=abs(correlations)
# fig,ax=plt.subplots(figsize=(20,20))
# sns.heatmap(correlations,vmax=1.0,center=0,fmt='.2f',square=True,linewidths=.5,
#             annot=True,cbar_kws={"shrink":.70})
# plt.show()




#Prepare Data
for k in range(len(Target)):
    X=data.drop(['%s'%Target[k]],axis=1)
    Y=data['%s'%Target[k]]
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
    X = scaler.transform(X)
    # Heat、Loading数据采用正则化
    normalizer = preprocessing.Normalizer(copy=True, norm='l2').fit(Y.values.reshape(1, -1))
    Y = normalizer.transform(Y.values.reshape(1, -1))
    Y = np.transpose(Y)


    #Divide Dataset
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=None)
    # print(X_train,X_test,Y_train,Y_test)


    #训练
    learning_rate=0.1
    training_epochs=500
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
        plt.figure(figsize=(10,10))
        plt.plot(X_train,Y_train,'ro',label='Original data')
        plt.plot(X_train,sess.run(W)*X_train+sess.run(b),label='Fitted line')
        plt.xlabel('%s'%Target[k])
        plt.ylabel('%s'%Target[k])
        # plt.legend()
        plt.savefig('F:/MOFS_Extra_Data/ML_Graph/Tensorflow_Graph/first%s.png'%Target[k])
        # plt.show()


        print("Testing...(Mean square loss Comparison")
        testing_cost=sess.run(tf.reduce_sum(tf.pow(pred-Y,2))/(2*X_test.shape[0]),feed_dict={X:X_test,Y:Y_test})
        print("Testing cost=",testing_cost)
        print("Absolute mean square loss difference:",abs(training_cost-testing_cost))
        plt.plot(X_test, Y_test, 'bo', label='Testing data')
        plt.plot(X_train, sess.run(W) * X_train + sess.run(b), label='Fitted line')
        plt.xlabel('%s' % Target[k])
        plt.ylabel('%s' % Target[k])
        # plt.legend()
        plt.savefig('F:/MOFS_Extra_Data/Tensorflow_picture/second%s.png' % Target[k])
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
