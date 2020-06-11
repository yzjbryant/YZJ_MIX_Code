from __future__ import print_function
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

Feature_Descriptor=['Cluster','Linker','Group','Length_a','Length_b','Length_c','Angle_alpha','Angle_beta',
                    'Angle_gamma','Number of Element','Tong','Zn','C','H','Br','Lu','I','N','O','F','S','Di','Df','Dif',
                    'Unitcell_volume','Density','ASA_A^2','ASA_m^2/cm^3','ASA_m^2/g','NASA_A^2','NASA_m^2/cm^3',
                    'NASA_m^2/g','AV_A^3','AV_Volume_fraction','AV_cm^3/g','NAV_A^3','NAV_Volume_fraction',
                    'NAV_cm^3/g','Energy_C4H4S_298K_10kpa','Energy_C4H4S_363K_10kpa','Energy_C4H4S_363K_100kpa',
                    'Energy_C6H6_298K_10kpa','Energy_C6H6_363K_10kpa','Energy_C6H6_363K_100kpa','Loading_C4H4S_298K_10kpa',
                    'Loading_C4H4S_363K_10kpa','Loading_C4H4S_363K_100kpa','Loading_C6H6_298K_10kpa','Loading_C6H6_363K_10kpa',
                    'Loading_C6H6_363K_100kpa']
X=data.drop(['Materials','Loading_C4H4S_363K_10kpa','Loading_C4H4S_363K_100kpa'],axis=1)
X=pd.get_dummies(X)
print(X.shape)
Y=data['Loading_C4H4S_363K_10kpa']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=None)

def normalize(X):
    mean=np.mean(X)
    std=np.std(X)
    X=(X-mean)/std
    return X
def append_bias_reshape(features,labels):
    m=features.shape[0]
    n=features.shape[1]
    x=np.reshape(np.c_[np.ones(m),features],[m,n+1])
    y=np.reshape(labels,[m,1])
    return x,y
# boston=tf.contrib.learn.datasets.load_dataset('boston')
# X_train,Y_train=boston.data,boston.target
# X_train=normalize(X_train)
# X_train,Y_train=append_bias_reshape(X_train,Y_train)
scaler=StandardScaler(copy=True,with_mean=True,with_std=True).fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
m=len(X_train)
n=48

X=tf.placeholder(tf.float32,name='X',shape=[m,n])
Y=tf.placeholder(tf.float32,name='Y')

w=tf.Variable(tf.random_normal([n,1]))
b=tf.Variable(tf.random_normal([n,1]))
Y_hat=tf.matmul(X,w)
loss=tf.reduce_mean(tf.square(Y-Y_hat,name='loss'))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
init_op=tf.global_variables_initializer()
total=[]
with tf.Session() as sess:
    sess.run(init_op)
    writer=tf.summary.FileWriter('graphs',sess.graph)
    for i in range(100):
        _,l=sess.run([optimizer,loss],feed_dict={X:X_train,Y:Y_train})
        total.append(l)
        print('Epoch {0}:Loss {1}'.format(i,1))
    writer.close()
    w_value,b_value=sess.run([w,b])
plt.plot(total)
plt.show()
N=500
X_new=X_train[N,:]
Y_pred=(np.matmul(X_new,w_value)+b_value).round(1)
print('Predicted value:${0} Actual value:/${1}'.format(Y_pred[33]*1000,Y_train[N]*1000,'\nDone'))