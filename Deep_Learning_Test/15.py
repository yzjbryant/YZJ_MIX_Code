#高深的深度学习最佳实践

#函数式API可以直接操作张量
from keras import Input,layers

input_tensor=Input(shape=(32,)) #一个张量
dense=layers.Dense(32,activation='relu')  #一个层是一个函数
output_tensor=dense(input_tensor) #可以在一个张量上调用一个层，它会返回一个张量

from keras.models import Sequential,Model
from keras import layers
from keras import Input

seq_model=Sequential()
seq_model.add(layers.Dense(32,activation='relu',input_shape=(64,)))
seq_model.add(layers.Dense(32,activation='relu'))
seq_model.add(layers.Dense(10,activation='softmax'))

#对应的函数式API实现
input_tensor=Input(shape=(64,))
x=layers.Dense(32,activation='relu')(input_tensor)
x=layers.Dense(32,activation='relu')(x)
output_tensor=layers.Dense(10,activation='softmax')(x)

model=Model(input_tensor,output_tensor)  #Model类将输入张量和输出张量转换一个模型
model.summary()

unrelated_input=Input(shape=(32,))
bad_model=model=Model(unrelated_input,output_tensor)

model.compile(optimizer='rmsprop',loss='catagorical_crossentropy')  #编译模型

import numpy as np
x_train=np.random.randint((1000,64))
y_train=np.random.randint((1000,64))

model.fit(x_train,y_train,epochs=10,batch_size=128)
score=model.evaluate(x_train,y_train)

