#解决聚类的问题
#聚成K类
#非监督学习
#人以类聚，物以群分
#K-means进行图像切割

#K-类，means-中心
#本质是：确定K类的中心点
#找到了中心点，就完成了聚类
#中心点-典型代表
#可自我纠正
#需设置初始的中心点，随机抽取
#近：欧几里得距离公式
#计算每个类的中心点：每个维度的平均值

#数据规范化：把值划分到[0,1]之间，百分率
#或者按照均值为0，方差为1的正态分布

#反复重复两个过程：
#1.确定中心点
#2.把其他的点按照距中心点的远近归到相应的中心点


# from sklearn.cluster import KMeans
#
# KMeans(n_clusters=8,    #K值
#        init='k-means++',#初始值选择方式
#        n_init=10,       #初始化中心点的运算次数
#        max_iter=300,
#        tol=0.0001,
#        precompute_distances='auto',
#        verbose=0,
#        random_state=None,
#        copy_x=True,
#        n_jobs=1,
#        algorithm='auto')#实现的算法
#
# ################20支亚洲球队聚类
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
#
# #输入数据
# data=pd.read_csv('data.csv',encoding='gbk')
# train_x=data[["2019年国际排名","2018世界杯","2015亚洲杯"]]
# df=pd.DataFrame(train_x)
# kmeans=KMeans(n_clusters=3)
#
# #规范化到[0,1]空间
# min_max_scaler=preprocessing.MinMaxScaler()
# train_x=min_max_scaler.fit_transform(train_x)
#
# #k-means算法
# kmeans.fit(train_x)
# predict_y=kmeans.predict(train_x)
#
# #合并聚类结果，插到源数据中
# result=pd.concat((data,pd.DataFrame(predict_y)),axis=1)
# result.rename({0:u'聚类'},axis=1,inplace=True)
# print(result)

############################Kmeans对图像进行分割
#加载图像，并对数据进行规范化
import Image
def load_data(filepath):
    #读文件
    f=open(filepath,'rb')
    data=[]
    #得到图像的像素值
    img=image.open(f)
    #得到图像尺寸
    width,height=img.size
    for x in range(width):
        for y in range(height):
            #得到点(x,y)的三个通道值
            c1,c2,c3=img.getpixel((x,y))
            data.append([c1,c2,c3])
    f.close()
    #采用Min_Max规范化
    mm=preprocessing.MinMaxScaler()
    data=mm.fit_transform(data)
    return np.mat(data),width,height

#加载图像，得到规范化的结果img，以及图像尺寸
img,width,height=load_data('./weixin.png')
#用K-means对图像进行2聚类
kmeans=KMeans(n_clusters=2)
kmeans.fit(img)
label=kmeans.predict(img)

#将图像聚类结果，转化成图像尺寸的矩阵
label=label.reshape([width,height])
#创建个新图像pic_mark，用来保存图像聚类的结果，并设置不同的灰度值
pic_mark=image.new("L",(width,height))
for x in range(width):
    for y in range(height):
        #根据类别设置图像灰度，类别0 灰度值255，类别1 灰度值127
        pic_mark.putpixel((x,y),int(256/(label[x][y]+1))-1)
pic_mark.save("weixin_mark.jpg","JPEG")
















