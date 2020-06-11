#Prinipal Component Analysis (PCA) 主成分分析
#无监督
#数据降维-数据预处理
#辅助手段
#减少需要分析的指标，尽可能多的保持数据的信息
#PCA并不是无损的
#数据压缩
#数据可视化和特征提取
#异常值检测和聚类
#铁路线就是主成分
#九门考试成绩，数学成绩就是主成分
#方差越大，信息量越多
#把数据看成空间上的点，进行投影，让这些点离得尽可能的远

#1.互不相关，协方差为0
#2.方差尽可能地大
#内积就是投影

#PCA要寻找一组基（主成分），互不相关，方差值尽可能的大
#协方差：两个变量之间的相关程度



#PCA过程
#1.得到数据，进行数据归一化
#2.得到协方差矩阵C
#3.求协方差矩阵的特征值和特征向量，并将特征值排序
#4.取前K个特征向量作为基，然后与X相乘得到降维之后的数据矩阵Y

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler

#IRIS
# #导入数据并进行归一化
# iris=load_iris()
# X=iris.data
#
# #X的归一化
# X_norm=StandardScaler().fit_transform(X)
# X_norm.mean(axis=0) #均值为0
#
# #PCA降维
# #1.求特征值和特征向量
# ew,ev=np.linalg.eig(np.cov(X_norm.T))
# #np.cov直接求协方差矩阵，每一行代表一个特征，每一轮代表样本
#
# #特征值特征向量排序
# ew_order=np.argsort(ew)[::-1]
# ew_sort=ew[ew_order]
# ev_sort=ev[:,ew_order] #ev的每一列代表一个特征向量
#
#
# #降成2维，然后取出排序后的特征向量的前两列就是基
# K=2
# V=ev_sort[:,:2]
#
# #得到降维后的数据
# X_new=X_norm.dot(V)
#
# #数据可视化
# colors=['red','black','orange']
# plt.figure()
# for i in [0,1,2]:
#     plt.scatter(X_new[iris.target==i,0],
#                 X_new[iris.target==i,1],
#                 alpha=0.7,
#                 c=colors[i],
#                 label=iris.target_names[i])
# plt.legend()
# plt.title('PCA of Iris')
# plt.xlabel('PC_0')
# plt.ylabel('PC_1')
# plt.show()
#
# #PCA一句话表达
# from sklearn.decomposition import PCA
#
# pca=PCA(n_components=2)
# X_new_1=pca.fit_transform(X_norm)


#人脸识别
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

#导入数据
faces=fetch_lfw_people(min_faces_per_person=60)
print(faces.images.shape)
print(faces.data.shape)

#可视化图片
fig,axes=plt.subplots(3,8,figsize=(8,4),subplot_kw={"xticks":[],"yticks":[]})
for i,ax in enumerate(axes.flat):
    ax.imshow(faces.images[i,:,:],cmap='gray')

#PCA降维
pca=PCA(150).fit(faces.data)
V=pca.components_
print(V.shape)












