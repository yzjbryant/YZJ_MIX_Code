#电影评论分类：二分类问题

from keras.datasets import imdb

#加载数据集
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

#准备数据,将整数序列编码为二进制矩阵
import numpy as np

def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros(len(sequences),dimension) #创建一个形状为(len(sequences),dimension)的零矩阵
    for i,sequences in enumerate(sequences):
        results[i,sequences] = 1. #将results[i]的指定索引设为1
    return results

x_train=vectorize_sequences(train_data) #将训练数据向量化
x_test=vectorize_sequences(test_data) #将测试数据向量化

#将标签向量化
y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

#构建网络 relu 整流线性单元
#sigmoid函数的输出值可看成概率值

#模型定义
from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#编译模型
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#使用自定义的损失和指标
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy()])

#留出验证集
x_val=x_train[:10000]
partial_x_train=x_train[10000:]

y_val=y_train[:10000]
partial_y_train=y_train[:10000]

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))

#绘制训练损失和验证损失
import matplotlib.pyplot as plt

history_dict=history.history
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']

epochs=range(1,len(loss_values)+1)

plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#绘制训练精度和验证精度
plt.clf()

acc=history_dict['acc']
val_acc=history_dict['val_acc']

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#使用训练好的网络在新数据上生成预测结果
model.predict(x_test)




































