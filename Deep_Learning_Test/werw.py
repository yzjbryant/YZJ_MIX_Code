import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import os

os.environ['TENSORFLOW_BACKEND']='tf'

data=np.random.random((1000,100))
labels=np.random.randint(2,size=(1000,1))
model=Sequential()
model.add(Dense(32,activation='relu',input_dim=100))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(data,labels,epochs=10,batch_size=32)
predictions=model.predict(data)

#数据：
#数据要存为Numpy数组或数组列表，使用sklearn.cross_validation的train_test_split模块进行分割为数据集和测试集

#Keras数据集:
from keras.datasets import boston_housing,mnist,cifar10,imdb

[x_train,y_train],(x_test,y_test)=mnist.load_data()
[x_train2,y_train2],(x_test2,y_test2)=boston_housing.load_data()
[x_train3,y_train3],(x_test3,y_test3)=cifar10.load_data()
[x_train4,y_train4],(x_test4,y_test4)=imdb.load_data(num_words=20000)
num_classes=10

#其他：
from urllib.request import urlopen
data=np.loadtxt(urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-databases/pima-indians-diabetes.data"),delimiter=",")
X=data[:,0:8]
y=data[:,8]

##预处理:
#序列填充：
from keras.preprocessing import sequence
x_train4=sequence.pad_sequences(x_train4,maxlen=80)
x_test4=sequence.pad_sequences(x_test4,maxlen=80)

#独热编码：
from keras.utils import to_categorical
Y_train=to_categorical(y_train,num_classes)
Y_test=to_categorical(y_test,num_classes)
Y_train3=to_categorical(y_train3,num_classes)
Y_test3=to_categorical(y_test3,num_classes)

#训练与测试集
from sklearn.model_selection import train_test_split
X_train5,X_test5,y_train5,y_test5=train_test_split(X,y,test_size=0.33,random_state=42)

#标准化/归一化
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(x_train2)
Standardized_X=scaler.transform(x_train2)
Standardized_X_test=scaler.transform(x_test2)

##模型架构
#序贯模型：
from keras.models import Sequential
model=Sequential()
model2=Sequential()
model3=Sequential()

#多层感知器
#二进制分类：
from keras.layers import Dense
model.add(Dense(12,input_dim=8,kernel_initializer='uniform',activation='relu'))
model.add(Dense(8,kernel_initializer='uniform',activation='relu'))
model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

#多级分类
from keras.layers import Dropout
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

#回归
model.add(Dense(64,activation='relu',input_dim=train_data.shape[1]))
model.add(Dense(1))

##卷积神经网络CNN
from keras.layers import Activation,Conv2D,MaxPooling2D,Flatten
model2.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))
model2.add(Activation('relu'))
model2.add(Conv2D(32,(3,3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))

model2.add(Conv2D(32,(3,3),padding='same'))
model2.add(Activation('relu'))
model2.add(Conv2D(32,(3,3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(512))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(num_classes))
model.add(Activation('softmax'))

##递归神经网络RNN
from keras.layers import Embedding,LSTM
model3.add(Embedding(20000,128))
model3.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))
model3.add(Dense(1,activation='sigmoid'))

##审视模型
model.output_shape #模型输出形状
model.summary()    #模型摘要展示
model.get_config() #模型配置
model.get_weights()#列出模型的所有权重张量


##编译模型
#多层感知器：二进制分类：
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#多层感知器：多级分类
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuraccy'])

#多层感知器：回归
model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

#递归神经网络：
model3.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

##模型训练
model3.fit(x_train4,y_train4,batch_size=32,epochs=15,verbose=1,validation_data=(x_test4,y_test4))

##评估模型
score=model3.evaluate(x_test,y_test,batch_size=32)

##预测
model3.predict(x_test4,batch_size=32)
model3.predict_classes(x_test4,batch_size=32)

#保存/加载模型
from keras.models import load_model
model3.save('modle_file.h5')
my_model=load_model('my_model.h5')

##模型微调
from keras.optimizers import RMSprop
opt=RMSprop(lr=0.01,decay=1e-6)
model2.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

##早停法
from keras.callbacks import EarlyStopping
early_stopping_monitor=EarlyStopping(patience=2)
model3.fit(x_train4,y_train4,batch_size=32,epochs=15,validation_data=(x_test4,y_test4),callbacks=[early_stopping_monitor])
