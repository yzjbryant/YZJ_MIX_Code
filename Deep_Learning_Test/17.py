import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features=2000
max_len=500

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)

x_train=sequence.pad_sequences(x_train,max_len)
x_test=sequence.pad_sequences(x_test,max_len)

model=keras.models.Sequential()
model.add(layers.Embedding(max_features,128,input_length=max_len,name='embed'))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

#为TensorBoard日志文件创建一个目录
#mkdir my_log_dir

#使用一个TensorBoard回调函数来训练模型
callbacks=[keras.callbacks.TensorBoard(log_dir='my_log_dir',histogram_freq=1,embeddings_freq=1)]
history=model.fit(x_train,y_train,epochs=20,batch_size=128,validation_split=0.2,callbacks=callbacks)

from keras.utils import plot_model
plot_model(model,show_shapes=True,to_file='model.png')











