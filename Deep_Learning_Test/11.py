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

#准备IMDB数据
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features=10000 #作为特征的单词个数
maxlen=500  #在这么多单词之后截断文本
batch_size=32

print('Loading data...')

(input_train,y_train),(input_test,y_test)=imdb.load_data(num_words=max_features)
print(len(input_train),'train sequences')
print(len(input_test),'train sequences')

print('Pad sequences(samples x time)')
input_train=sequence.pad_sequences(input_train,maxlen=maxlen)
input_test=sequence.pad_sequences(input_test,maxlen=maxlen)
print('input_train shape:',input_train.shape)
print('input_test shape:',input_test.shape)

#用Embedding层和SimpleRNN层来训练模型

from keras.layers import Dense

model=Sequential()
model.add(Embedding(max_features,32))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(input_train,y_train,epochs=10,batch_size=128,validation_split=0.2)

#绘制结果
import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()


#LSTM长短期记忆
#LSTM架构的详细为代码

output_t=activation(dot(state_t,Uo)+dot(input_t,Wo)+dot(C_t,Vo)+bo)

i_t=activation(dot(state_t,Ui)+dot(input_t,Wi)+bi)
f_t=activation(dot(state_t,Uf)+dot(input_t,Wf)+bf)
k_t=activation(dot(state_t,Uk)+dot(input_t,Wk)+bk)

c_t+1=i_t*k_t+c_t*f_t

from keras.layers import LSTM

model=Sequential()
model.add(Embedding(max_features,32))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(input_train,y_train,epochs=10,batch_size=128,validation_split=0.2)




























