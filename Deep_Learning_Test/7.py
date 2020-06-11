#深度学习用于计算机视觉
#猫狗分类

#将图像复制到训练、验证和测试的目录

import os,shutil

original_dataset_dir='Users/fchollet/Downloads/kaggle_original_data' #原始数据集解压目录的路径

base_dir='Users/fchollet/Downloads/cats_and_dogs_small' #保存较小的数据集
os.mkdir(base_dir)

train_dir=os.path.join(base_dir,'train')
os.mkdir(train_dir)

validation_dir=os.path.join(base_dir,'validation')
os.mkdir(validation_dir)

test_dir=os.path.join(base_dir,'test')
os.mkdir(test_dir)
#分别对应划分后的训练、验证和测试的目录

#猫的训练图像目录
train_cats_dirs=os.path.join(train_dir,'cats')
os.mkdir(train_cats_dir)
#狗的训练图像目录
train_dogs_dirs=os.path.join(train_dir,'dogs')
os.mkdir(train_dogs_dir)


#猫的验证图像目录
validation_cats_dir=os.path.join(validation_dir,'cats')
os.mkdir(validation_cats_dir)
#狗的验证图像目录
validation_dogs_dir=os.path.join(validation_dir,'dogs')
os.mkdir(validation_dogs_dir)


#猫的测试图像目录
test_cats_dir=os.path.join(test_dir,'cats')
os.mkdir(test_cats_dir)
#狗的测试图像目录
test_dogs_dir=os.path.join(test_dir,'dogs')
os.mkdir(test_dogs_dir)

#将前1000张猫的图像复制到train_cats_dir
fnames=['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)
    dst=os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dst)

#将接下来500张猫的图像复制到validation_cats_dir
fnames=['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)
    dst=os.path.join(validation_cats_dir,fname)
    shutil.copyfile(src,dst)

#将接下来500张猫的图像复制到test_cats_dir
fnames=['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)
    dst=os.path.join(test_cats_dir,fname)
    shutil.copyfile(src,dst)



# 将前1000张狗的图像复制到train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 将接下来500张狗的图像复制到validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

#将接下来500张狗的图像复制到test_cats_dir
fnames=['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)
    dst=os.path.join(test_dogs_dir,fname)
    shutil.copyfile(src,dst)

#检查每个分组中分别包含多少张图像
print('total training cat images:',len(os.listdir(train_cats_dir)))
print('total training dog images:',len(os.listdir(train_dogs_dir)))

print('total validation cat images:',len(os.listdir(validation_cats_dir)))
print('total validation dog images:',len(os.listdir(validation_dogs_dir)))

print('total test cat images:',len(os.listdir(test_cats_dir)))
print('total test dog images:',len(os.listdir(test_dogs_dir)))
#每个分组中两个类别的样本数相同，是平衡二分类问题，分类精度可作为衡量成功的指标

#构建网络

#将猫狗分类的小型卷积神经网络实例化
from keras import layers
from keras import models

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#看特征图的维度如何随着每层变化
model.summary()

#在编译这一步，使用RMSprop优化器，用二元交叉熵作为损失函数

from keras import optimizers

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

#数据预处理
#将数据格式化为经过预处理的浮点数张量

#1.读取图像文件
#2.将JPEG文件解码为RGB像素网格
#3.将这些像素网格转换为浮点数张量
#4.将像素值(0~255)缩放到[0,1]区间 （神经网络喜欢处理较小的输入值）

#使用ImageDataGenerator从目录中读取图像
from keras.preprocessing.image import ImageDataGenerator

#将所有图像乘以1/255缩放
train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train_dir,#目标目录
                                                  target_size=(150,150),#将所有图像调整为150*150
                                                  batch_size=20,#
                                                  class_mode='binary'#因为使用了binary_crossentropy损失，所以需要用二进制标签)
validation_generator=test_datagen.flow_from_directory(validation_dir,
                                                      target_size=(150,150),
                                                      batch_size=20,
                                                      class_mode='binary')

#理解python生成器
def generator():
    i=0
    while True:
        i += 1
        yield i
for item in generator():
    print(item)
    if item>4:
        break

for data_batch,labels_batch in train_generator:
    print('data batch shape:',data_batch.shape)
    print('labels batch shape:',labels_batch.shape)
    break


#利用批量生成器拟合模型
history=model.fit_generator(train_generator,
                            steps_per_epoch=100,
                            epochs=30,
                            validation_data=validation_generator,
                            validation_steps=50)

#保存模型
model.save('cats_and_dogs_small_1.h5')

#绘制训练过程中的损失函数和精度曲线
import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#使用数据增强

#利用ImageDataGenerator来设置数据增强
datagen=ImageDataGenerator(rotation_range=40, #图像随机旋转角度值
                           width_shift_range=0.2, #图像在水平或垂直方向上平移的范围
                           height_shift_range=0.2,
                           shear_range=0.2, #随机错切变换的角度
                           zoom_range=0.2,  #图像随机缩放的范围
                           horizontal_flip=True, #随机将一半图像水平翻转
                           fill_mode='nearest' #填充新创建像素)

#显示几个随机增强后的训练图像
from keras.preprocessing import image #图像预处理工具的模块

fnames=[os.path.join(train_cats_dirs,fname) for fname in os.listdir(train_cats_dirs)]
img_path=fnames[3] #选择一张图像进行增强
img=image.load_img(img_path,target_size=(150,150)) #读取图像并调整大小
x=image.img_to_array(img) #将其转换为形状(150,150,3)的Numpy数组
x=x.reshape((1,) + x.shape) #将其形状改变为(1,150,150,3)

#生成随机变换后的图像批量，循环是无限的，因此你需要在某个时刻终止循环
i=0
for batch in datagen.flow(x,batch_size=1):
    plt.figure(i)
    imgplot=plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i%4 ==0:
        break
plt.show()

#定义一个包含dropout的新卷积神经网络
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5)) #＋
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

#利用数据增强生成器训练卷积神经网络
train_datagen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255) #注意，不能增强验证数据

train_generator=train_datagen.flow_from_directory(train_dir, #目标目录
                                                  target_size=(150,150), #将所有图像的大小调整为150*150
                                                  batch_size=32,
                                                  class_mode='binary' #因为使用了binary_crossentropy损失，所有需要用二进制标签)

validation_generator=test_datagen.flow_from_directory(validation_dir,
                                                      target_size=(150,150),
                                                      batch_size=32,
                                                      class_mode='binary')

history=model.fit_generator(train_generator,
                            steps_per_epoch=100,
                            epochs=100,
                            validation_data=validation_generator,
                            validation_steps=50)

#保存模型
model.save('cats_and_dogs_small_2.h5')






#使用预训练的卷积神经网络
#VGG16架构
#使用预训练网络有两种方法：特征提取、微调模型
#卷积基：池化层+卷积层+密集连接层
#特征提取就是取出之前训练好的网络的卷积基，在上面运行新数据，然后在输出上面训练一个新的分类器




#使用在ImageNet上训练的VGG16网络中的卷积基从猫狗图像中提取有趣的特征，然后在这些特征上训练一个猫狗分类器
#我们将VGG16模型实例化

from keras.applications import VGG16
conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3)))
#VGG16卷积基架构
conv_base.summary()

#使用预训练的卷积基提取特征
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir='Users/fchollet/Downloads/cats_and_dogs_small'
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'test')

datagen=ImageDataGenerator(rescale=1./255)
batch_size=20

def extract_features(directory,sample_count):
    features=np.zeros(shape=(sample_count,4,4,512))
    labels=np.zeros(shape=(sample_count))
    generator=datagen.flow_from_directory(directory,target_size=(150,150),
                                          batch_size=batch_size,class_mode='binary')
    i=0
    for inputs_batch,labels_batch in generator():
        features_batch=conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size]=features_batch
        labels[i*batch_size:(i+1)*batch_size]=labels_batch
        i += 1
        if i*batch_size>=sample_count:
            break  #注意：这些生成器在循环中不断生成数据，所以你必须在读取完所有图像后终止循环
    return features,labels


train_features=np.reshape(train_features,(2000,4*4*512))
validation_features=np.reshape(validation_features,(1000,4*4*512))
test_features=np.reshape(test_features,(1000,4*4*512))

#定义并训练密集连接分类器
from keras import models
from keras import layers
from keras import optimizers

model=models.Sequential()
model.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history=model.fit(train_features,train_labels,epochs=30,batch_size=20,
                  validation_data=(validation_features,validation_labels))

#绘制结果
import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Valiadtion accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and Valiadtion loss')
plt.legend()

plt.show()





##使用数据增强的特征提取
#扩展conv_base模型，然后从输入数据上端到端地运行模型
#计算代价太高，要在GPU上运行

#在卷积基上添加一个密集连接分类器
from keras import models
from keras import layers

model=models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.summary()

#在编译和训练模型之前，一定要冻结卷积基，冻结一个或多个层是指在训练过程中保持权重不变
#将其trainable属性设为False  冻结
print('This is the number of trainable weights''before freezing the conv base:',
      len(model.trainable_weights))
conv_base.trainable=False
print('This is the number of trainable weights''before freezing the conv base:',
      len(model.trainable_weights))

#利用冻结的卷积基端到端地训练模型
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')
test_datagen=ImageDataGenerator(rescale=1./255) #注意：不能增强验证数据

train_generator=train_datagen.flow_from_directory(train_dir,
                                                  target_size=(150,150),
                                                  batch_size=20,
                                                  class_mode='binary')
validation_generator=test_datagen.flow_from_directory(validation_dir,
                                                      target_size=(150,150),
                                                      batch_size=20,
                                                      class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,
                            validation_data=validation_generator,
                            validation_steps=50)


#绘制结果
import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Valiadtion accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and Valiadtion loss')
plt.legend()

plt.show()



###微调模型
conv_base.summary()

#冻结直到某一层地所有层
conv_base.trainable=True
set_trainable=False
for layer in conv_base.layers:
    if layer.name=='block5_conv1':
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else:
        layer.trainable=False

#微调模型
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=100,
                            validation_data=validation_generator,
                            validation_steps=50)


#绘制结果
import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Valiadtion accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and Valiadtion loss')
plt.legend()

plt.show()

#使曲线变得平滑
def smooth_curve(points,factor=0.8):
    smoothed_points=[]
    for point in points:
        if smoothed_points:
            previous=smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs,smooth_curve(acc),'bo',labels_batch='Smoothed training acc')
plt.plot(epochs,smooth_curve(val_acc),'bo',labels_batch='Smoothed validation acc')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,smooth_curve(loss),'bo',labels_batch='Smoothed training loss')
plt.plot(epochs,smooth_curve(val_loss),'bo',labels_batch='Smoothed validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()



#在测试数据上最终评估模型
test_generator=test_datagen.flow_from_directory(test_dir,
                                                target_size=(150,150),
                                                batch_size=20,
                                                class_mode='binary')
test_loss,test_acc=model.evaluate_generator(test_generator,steps=50)
print('test acc',test_acc)









































































































































































