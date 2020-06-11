#AdaBoost - 三个臭皮匠，顶个诸葛亮
#集思广益 集成的含义

#集成算法有两种
#1.投票选举：bagging - 少数服从多数 - 提升 - 对前面的结果进行优化
#2.再学习：boosting - 加权融合 - 相互独立，不存在独立性

#Adaptive Boosting - 再学习
#自适应提升算法

#Boosting:训练多个弱分类器，根据权重组成强分类器
#表现好，权重高

#计算权重：通过误差率
#得到最优弱分类器：改变样本的权重或数据分布

#过程：
#确定初始样本的权重，然后训练分类器，根据误差最小，选择分类器，得到误差率，计算该分类器的权重
#然后根据该分类器的误差去重新计算样本的权重
#进行下一轮的训练，若不停止，重复上述过程

#实战：预测房价
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import AdaBoostRegressor
#
# AdaBoostClassifier(base_estimator=None,#弱分类器
#                    n_estimators=50,#最大迭代次数
#                    learning_rate=1.0,#学习率
#                    algorithm='SAMME.R',#算法
#                    random_state=None)#随机数种子
#
# AdaBoostRegressor(base_estimator=None,#弱分类器
#                    n_estimators=50,#最大迭代次数
#                    learning_rate=1.0,#学习率
#                    loss='linear',
#                    random_state=None)#随机数种子

#波士顿房价预测
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import load_boston
# from sklearn.ensemble import AdaBoostRegressor
#
# #加载数据
# data=load_boston()
# #分割数据
# train_x,test_x,train_y,test_y=train_test_split(
#     data.data,data.target,test_size=0.25,random_state=33)
# #AdaBoost回归模型
# regressor=AdaBoostRegressor()
# regressor.fit(train_x,train_y)
# pred_y=regressor.predict(test_x)
# mse=mean_squared_error(test_y,pred_y)
# print("房价预测结果",pred_y)
# print("均方误差",round(mse,2))

#########AdaBoost与决策树模型的比较
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

#设置AdaBoost迭代次数
n_estimator=200
X,y=datasets.make_hastie_10_2(n_samples=12000,random_state=1)
train_x,train_y=X[2000:],y[2000:]
test_x,test_y=X[:2000],y[:2000]

#弱分类器
dt_stump=DecisionTreeClassifier(max_depth=1,min_samples_leaf=1)
dt_stump.fit(train_x,train_y)
dt_stump_err=1.0-dt_stump.score(test_x,test_y)

#决策树分类器
dt=DecisionTreeClassifier()
dt.fit(train_x,train_y)
dt_err=1.0-dt.score(test_x,test_y)

#AdaBoost分类器
ada=AdaBoostClassifier(base_estimator=dt_stump,n_estimators=n_estimator)
ada.fit(train_x,train_y)


#可视化错误率
fig=plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']
ax=fig.add_subplot(111)
ax.plot([1,n_estimator],[dt_stump_err]*2,'k-',label=u'决策树弱分类器 错误率')
ax.plot([1,n_estimator],[dt_err]*2,'k--',label=u'决策树模型 错误率')
ada_err=np.zeros((n_estimator,))

for i,pred_y in enumerate(ada.staged_predict(test_x)):
    ada_err[i]=zero_one_loss(pred_y,test_y)
ax.plot(np.arange(n_estimator)+1,ada_err,label='Ada')
ax.set_xlabel('迭代次数')
ax.set_ylabel('错误率')
leg=ax.legend(loc='upper right',fancybox=True)
plt.show()











