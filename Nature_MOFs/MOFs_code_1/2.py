#_*_coding:utf-8_*_

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#归一化函数
def normalize(X):
    mean=np.mean(X)
    std=np.std(X)
    X=(X-mean)/std
    return X

#固定输入值将权重和偏置结合起来
def append_bias_reshape(features,labels):
    m=features.shape[0]
    n=features.shape[1]
    x=np.reshape(np.c_[np.ones(m),features],[m,n+1])
    y=np.reshape(labels,[m,1])
    return x,y


#数据加载及处理
boston=tf.contrib.learn.datasets.load_dataset('boston')
X_train,Y_train=boston.data,boston.target
X_train=normalize(X_train)
X_train,Y_train=append_bias_reshape(X_train,Y_train)

m=len(X_train)    #X数据集训练数目
n=13+1
print(m)
X=tf.placeholder(tf.float32,name='X',shape=[m,n])
Y=tf.placeholder(tf.float32,name="Y")

#为权重和偏置创建TensorFlow变量，通过随机数初始化权重
# b=tf.Variable(tf.random_normal([m,n]))
w=tf.Variable(tf.random_normal([n,1]))

#定义要用于预测的线性回归模型，现在需要矩阵乘法来完成这个任务
Y_hat=tf.matmul(X,w)

#损失函数
loss=tf.reduce_mean(tf.square(Y-Y_hat),name='loss')

#优化器
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

#初始化操作符
init_op=tf.global_variables_initializer()
total=[]

#开始计算图
with tf.Session() as sess:
    sess.run(init_op)
    writer=tf.summary.FileWriter('graphs',sess.graph)
    for i in range(100):
        _,l=sess.run([optimizer,loss],feed_dict={X:X_train,Y:Y_train})
        total.append(l)
        print("Epoch {0} : Loss {1} .".format(i,l))
    writer.close()
    w_value=sess.run([w])

plt.plot(total)
plt.show()