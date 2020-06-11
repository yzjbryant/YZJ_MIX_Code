#二分类模型
#SVM的目的是为了寻找一个超平面-决策面
#鲁棒性：容错能力的强弱
#Special feature：分类间隔
#极限位置到最优决策面C之间的距离-分类间隔(margin)
#最大间隔为SVM的最优解
#超平面为线性函数，g(x)=(w^T)*x+b
#w为法向量，决定了超平面的方向
#支持向量：两个点决定的支撑超平面
#具体求解是一个凸优化的问题

#完全线性可分：硬间隔最大化
#不完全线性可分：软间隔最大化


#非线性支持向量机
#需引入核函数
#一拍桌子，球飞起来
#拍的过程就称为核函数映射的过程

##https://mp.weixin.qq.com/s/HtxCeXeZ_pZlAPvq8L_bfQ

from sklearn import svm
#SVR：回归，SV-Regression 只能使用线性核函数
#SVC: 分类，SC-Classification 针对非线性数据

#C:惩罚系数越大，分类器准确性高，容错率低，泛化率差
#C:越小，泛化能力强，准确率降低
#model.fit(train_X,train_Y)
#model.predict(test_X)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#加载数据集
data=pd.read_csv("./data.csv")
pd.set_option('display.max_columns',None)
print(data.columns)
print(data.head(5))
print(data.describe())

#数据清洗
features_mean=list(data.columns[2:12])
features_se=list(data.columns[12,22])
features_worst=list(data.columns[22:32])
data.drop("id",axis=1,inplace=True)
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

#可视化肿瘤诊断结果
sns.countplot(data['diagnosis'],label="Count")
plt.show()
#用热力图
corr=data[features_mean].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,annot=True)
plt.show()

#特征选择
features_remain=['1','2','3','4']
train,test=train_test_split(data,test_size=0.3)
train_X=train[features_remain]
train_y=train['diagnosis']
test_X=test[features_remain]
test_y=test['diagnosis']

#采用Z-Score规范化数据，保证每个特征维度的
#数据均值为0，方差为1
ss=StandardScaler()
train_X=ss.fit_transform(train_X)
test_X=ss.transform(test_X)

#创建SVM分类器
model=svm.SVC()
#训练
model.fit(train_X,train_y)
#预测
prediction=model.predict(test_X)
print(metrics.accuracy_score(prediction,test_y))

























