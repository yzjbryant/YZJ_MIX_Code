#RNN 伪代码

state_t=0 #t时刻地状态
for input_t in input_sequence:  #对序列元素进行遍历
    output_t=f(input_t,state_t)
    state_t=output_t #前一次地输出变成下一次迭代的状态


#更详细的RNN伪代码

state_t=0
for input_t in input_sequence:
    output_t=activation(dot(W,input_t) + dot(U,state_t)+b)
    state_t=output_t

#简单的RNN的Numpy实现
import numpy as np

timesteps=100
input_features=32
output_features=64

#输入数据：随机噪声，仅作为示例
inputs=np.random.random((timesteps,input_features))

state_t=np.zeros((output_features))  #初始状态：全零向量

W=np.random.random((output_features,input_features))
U=np.random.random((output_features,output_features))
b=np.random.random((output_features,))

successive_outputs=[]
for input_t in inputs:
    output_t=np.tanh(np.dot(W,input_t) + np.dot(U,state_t)+b)
    state_t=output_t
final_output_sequence=np.stack(successive_outputs,axis=0)


#Keras中的循环层
from keras.layers import SimpleRNN

from keras.models import Sequential
from keras.layers import SimpleRNN,Embedding

model=Sequential()
model.add(Embedding(10000,32))
model.add(SimpleRNN(32,return_sequences=True))
model.add(SimpleRNN(32,return_sequences=True))
model.add(SimpleRNN(32,return_sequences=True))
model.add(SimpleRNN(32))
model.summary()
































