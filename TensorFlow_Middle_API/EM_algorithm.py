#EM聚类 含有隐变量的概率模型参数的极大似然估计法
#Expectiation Maximization
#最大期望算法
#一份菜，多次匀，分成最终两份
#（初始化参数-观察预期）Expectiation步
#（重新估计参数） Maximization步
#最大似然 - Maximum Likelihood
#最大似然就是最大可能性的意思 - 男同学高的可能性要大
#最大似然估计- 已知一个人高于另一个人，反推那个人是男的
#最大似然估计-通过已知结果，估计参数
#EM算法就是求解最大似然估计的方法

#E步：通过旧的参数来计算隐藏变量
#M步：通过得到的隐藏变量的结果来重新估计参数
#直到参数不发生变化

#无监督模型-聚类
#软聚类
#GMM高斯混合模型-概率密度，正态分布
#HMM隐马尔可夫模型-NLP

#######实战：王者荣耀英雄的聚类
# from sklearn.mixture import GaussianMixture
# gmm=GaussianMixture(n_components=1,#要聚类的个数
#                     covariance_type='full',#协方差类型
#                     max_iter=100)#最大迭代次数

import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

data_ori=pd.read_csv('data.csv',encoding='gb18030')
features=[u'1',u'2']
data=data_ori[features]

#热力图
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
corr=data[features].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,annot=True)
plt.show()

#特征工程 降维
features_remain=[u'1',u'2']
data=data_ori[features_remain]

data[u'1']=data[u'1'].apply(lambda x:float(x.strip('%'))/100)
data[u'1']=data[u'2'].map({'远程':1,'近战':2})

ss=StandardScaler()
data=ss.fit_transform(data)



##gmm
gmm=GaussianMixture(n_components=30,covariance_type='full')
gmm.fit(data)

prediction=gmm.predict(data)
print(prediction)

data_ori.insert(0,'分组',prediction)

#评估
from sklearn.metrics import calinski_harabaz_score
print(calinski_harabaz_score(data,prediction))

#指标分数越高，聚类效果越好
#相同类中差异性小，不同类间差异性大
























