#MINI-BATCH 完整代码

import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from datetime import datetime
import time


now=datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir="tf_logs"
logdir="{}/run-{}/".format(root_logdir,now)
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

housing=fetch_california_housing()
m,n=housing.data.shape
print(housing.data)
print("数据集：{}行，{}列".format(m,n))
housing_data_plus_bias=np.c_[np.ones((m,1)),housing.data]  #以列的维度创建堆叠数组
scaler=StandardScaler()
scaled_housing_data=scaler.fit_transform(housing.data)
# print(scaled_housing_data)
scaled_housing_data_plus_bias=np.c_[np.ones((m,1)),scaled_housing_data]
# print(scaled_housing_data_plus_bias)

n_epochs=1000
learning_rate=0.01

X=tf.placeholder(tf.float32,shape=(None,n+1),name="X")
y=tf.placeholder(tf.float32,shape=(None,1),name="y")
theta=tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0,seed=42),name="theta")
y_pred=tf.matmul(X,theta,name="predictions")
error=y_pred-y
mse=tf.reduce_mean(tf.square(error),name="mse")
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(mse)

init=tf.global_variables_initializer()

#保存和恢复模型
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch",epoch,"MSE=",mse.eval())
            save_path=saver.save(sess,"/tmp/my_model.ckpt")
        sess.run(training_op)
    best_theta=theta.eval()
    save_path=saver.save(sess,"/tmp/my_model_final.ckpt")


n_epochs=10
batch_size=100
n_batches=int(np.ceil(m/batch_size))

def fetch_batch(epoch,batch_index,batch_size):
    know=np.random.seed(epoch*n_batches + batch_index)
    print("我是know:",know)
    indices=np.random.randint(m,size=batch_size)
    X_batch=scaled_housing_data_plus_bias[indices]
    y_batch=housing.target.reshape(-1,1)[indices]
    return X_batch,y_batch

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch,y_batch=fetch_batch(epoch,batch_index,batch_size)
            if batch_size%10==0:
                summary_str=mse_summary.eval(feed)
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})

    best_theta=theta.eval()
print(best_theta)

#使用Tensorboard展示图形和训练曲线
mse_summary=tf.summary.scalar('MSE',mse)
file_writer=tf.summary.FileWriter(logdir,tf.get_default_graph())
file_writer.close()






























