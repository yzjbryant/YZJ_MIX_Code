#深度学习用于计算机视觉

#卷积神经网络 Convnet

from keras import layers
from keras import models

#实例化一个小型的卷积神经网络

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

#在卷积神经网络上添加分类器

model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

#在MNIST图像上训练卷积神经网络

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

train_images=train_images.reshape((60000,28,28,1))
train_images=train_images.astype('float32') / 255

test_images=test_images.reshape((10000,28,28,1))
test_images=test_images.astype('float32') /255

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=5,batch_size=64)

#在测试数据上对模型进行评估

test_loss,test_acc=model.evaluate(test_images,test_labels)
print(test_acc)













































