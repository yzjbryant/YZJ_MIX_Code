import tensorflow as tf

#打印时间分割线
@tf.function
def printbar():
    today_ts=tf.timestamp()%(24*60*60)
    hour=tf.cast(today_ts//3600+8,tf.int32)%tf.constant(24)
    minute=tf.cast((today_ts%3600)//60,tf.int32)
    second=tf.cast((tf.floor(today_ts%60),tf.int32))

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}",m))==1:
            return (tf.strings.format("O{}",m))
        else:
            return (tf.string.format("{}",m))

    timestring=tf.string.join([timeformat(hour),timeformat(minute),timeformat(second)],separator=":")
    tf.print("=============="*8+timestring)

#Linear Regression Models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#样本数量
n=400
#生成测试用数据集
X=tf.random.uniform([n,2],minval=-10,maxval=10)
w0=tf.constant([[2,0],[-3,0]])
b0=tf.constant([[3,0]])
Y=X@w0+b0+tf.random.normal([n,1],mean=0.0,stddev=2.0)  #表示矩阵乘法，增加正态扰动

#数据可视化

plt.figure(figsize=(12,5))
ax1=plt.subplot(121)
ax1.scatter(X[:,0],Y[:,0],c="b")
plt.xlabel("x1")
plt.ylabel("y",rotation=0)


ax2=plt.subplot(122)
ax2.scatter(X[:,1],Y[:,0],c="g")
plt.xlabel("x2")
plt.ylabel("y",rotation=0)
plt.show()

#构建数据管道迭代器
def data_iter(features,labels,batch_size=8):
    num_examples=len(features)
    indices=list(range(num_examples))
    np.random.shuffle(indices)#样本的读取顺序是随机的
    for i in range(0,num_examples,batch_size):
        indexs=indices[i:min(i+batch_size,num_examples)]
        yield tf.gather(X,indexs),tf.gather(Y,indexs)

#测试数据管道效果
batch_size=8
(features,labels)=next(data_iter(X,Y,batch_size))
print(features)
print(labels)

##定义模型
w=tf.Variable(tf.random.normal(w0.shape))
b=tf.Variable(tf.zeros_like(b0,dtype=tf.float32))

#定义模型
class LinearRegression:
    #正向传播
    def __call__(self, x):
        return x@w+b
    #损失函数
    def loss_func(self,y_true,y_pred):
        return tf.reduce_mean((y_true-y_pred)**2/2)

model=LinearRegression()

##训练模型
#使用动态图调试
def train_step(model,features,labels):
    with tf.GradientTape() as tape:
        predictions=model(features)
        loss=model.loss_func(labels,predictions)
    #反向传播求梯度
    dloss_dw,dloss_db=tape.gradient(loss,[w,b])
    #梯度下降法更新参数
    w.assign(w-0.001*dloss_dw)
    b.assign(b-0.001*dloss_db)

    return loss
def train_model(model,epochs):
    for epoch in tf.range(1,epochs+1):
        for features,labels in data_iter(X,Y,10):
            loss=train_step(model,features,labels)
        if epoch%50==0:
            printbar()
            tf.print("epoch=",epoch,"loss=",loss)
            tf.print("w=",w)
            tf.print("b=",b)
train_model(model,epochs=200)



#测试train_step效果
batch_size=10
(features,labels)=next(data_iter(X,Y,batch_size))
train_step(model,features,labels)


