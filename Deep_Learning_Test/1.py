from keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

print(train_images.shape)
print(len(train_labels))
print(train_labels)

print(test_images.shape)
print(len(test_labels))
print(test_labels)

from keras import models
from keras import layers


#全连接层
network=models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
#output=relu(dot(W,input) + b)
network.add(layers.Dense(10,activation='softmax'))

#编译(compile)
network.compile(optimizer='rmsprop',loss='categorical_crossentropy')

#准备图像数据
train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255

test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255

#准备标签
from keras.utils import to_categorical

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

#fit 拟合
network.fit(train_images,train_labels,epochs=5,batch_size=128)

#检查模型在测试集上的性能
test_loss,test_acc=network.evaluate(test_images,test_labels)
print('test_acc:',test_acc)

#张量(tensor):数字的容器
#张量的维度(dimension)通常叫做轴(axis)

digit=train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit,cmap=plt.cm.binary)
plt.show


def naive_relu(x):
    assert len(x.shape) == 2

    x=x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j]=max(x[i,j],0)
    return x

def add_relu(x):
    assert len(x.shape) == 2
    assert x.shape==y.shape

    x=x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] += y[i,j]
    return x
import numpy as np
z=x+y
z=np.maximum(z,0.)

#广播
def naive_add_matrix_and_vector(x,y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1]==y.shape[0]

    x=x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] += y[j]
    return x

import numpy as np
z=np.dot(x,y)










































































