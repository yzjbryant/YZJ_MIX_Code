#卷积神经网络的可视化

#1.可视化中间激活
from keras.models import load_model
model=load_model('cats_and_dogs_small_2.h5')
model.summary() #作为提醒

#预处理单张图像
img_path='/Users/fchollet/Downloads/cats_and_dogs_small/test/cats/cat.1700.jpg'

from keras.preprocessing import image #将图像预处理为一个4D张量
import numpy as np

img=image.load_img(img_path,target_size=(150,150))
img_tensor=image.img_to_array(img)
img_tensor=np.expand_dims(img_tensor,axis=0)
img_tensor /= 255. #请记住，训练模型的输入数据都用这种方法预处理

#其形状为(1,150,150,3)
print(img_tensor.shape)

#显示测试图像
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

#用一个输入张量和一个输出张量列表将模型实例化
from keras import models

layer_outputs=[layer.output for layer in model.layers[:8]]  #提取前8层的输出
activation_model=model.Model(inputs=model.input,outputs=layer_outputs)
#创建一个模型，给定模型输入，可以返回这些输出

#以预测模式运行模型
activations=activation_model.predict(img_tensor)
#返回8个Numpy数组组成的列表，每个层激活对应一个Numpy数组

#对于输入的猫图像，第一个卷积层的激活如下所示：
first_layer_activation=activations[0]
print(first_layer_activation)


#将第4个通道可视化
import matplotlib.pyplot as plt

plt.matshow(first_layer_activation[0,:,:,4],cmap='viridis')

#将第7个通道可视化
plt.matshow(first_layer_activation[0,:,:,7],cmap='viridis')


#将每个中间激活的所有通道可视化
layer_names=[]
for layer in model.layers[:8]:
    layer_names.append(layer_name)  #层的名称，这样你可以将这些名称画到图中

images_per_row=16
for layer_name,layer_activation in zip(layer_names,activations):   #显示特征图
    n_features=layer_activation.shape[-1]  #特征图中的特征个数

    size=layer_activation.shape[1]  #特征图的形状为(1,size,size,n_features)
    n_cols=n_features // images_per_row #在这个矩阵中将激活通道平铺
    display_grid=np.zeros((size*n_cols,images_per_row*size))

    for col in range(n_cols):   #将每个过滤器平铺到一个大的水平网格中
        for row in range(images_per_row):
            channel_image=layer_activation[0,:,:,col*images_per_row+row]

            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image =np.clip(channel_image,0,255).astype('uint8')
            display_grid[col*size:(col+1)*size,
                            row*size:(row+1)*size]=channel_image  #显示网格
            scale = 1./size
            plt.figure(figure=(scale*display_grid.shape[1],
                               scale.display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid,aspect='auto',cmap='viridis')


#2.可视化卷积神经网络的过滤器

#为过滤器的可视化定义损失张量
from keras.applications import VGG16
from keras import backend as K

model=VGG16(weights='imagenet',include_top=False)
layer_name='block3_conv1'
filter_index=0

layer_output=model.get_layer(layer_name).output
loss=K.mean(layer_output[:,:,:,filter_index])

#获取损失相对于输入的梯度
grads=K.gradients(loss,model.input)[0]
#调用gradients返回的是一个张量列表（本例中列表长度为1）。因此，只保留第一个元素，它是一个张量

#梯度标准化的技巧
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
#做除法前加上1e-5,以防不小心除以0

#给定Numpy输入值，得到Numpy输出值
iterate=K.function([model.input],[loss,grads])
import numpy as np
loss_value,grads_value=iterate([np.zeros((1,150,150,3))])

#通过随机梯度下降让损失最大化
input_img_data=np.random.random((1,150,150,3)) *20 +128. #从一张带有噪声的灰度图像开始
step=1. #每次梯度更新的步长
for i in range(40):
    loss_value,grads_value=iterate([input_img_data]) #计算损失值和梯度值
    input_img_data += grads_value*step #沿着让损失最大化的方向调节输入图像
    #运行40次梯度上升

#将张量转换为有效图像的实用函数

#可视化类激活的热力图

#加载带有预训练权重的VGG16网络
#为VGG16模型预处理一张输入图像
#应用Grad-CAM算法
#热力图后处理

from keras.applications.vgg16 import VGG16
model=VGG16(weights='imagenet')

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np

img_path='/Users/fchollet/Downloads'



#将热力图与原始图叠加
import cv2

img=cv2.imread(img_path)
heatmap=cv2.resize(heatmap,(img.shape[1],img.shape[0]))
heatmap=np.uint8(255*heatmap)
heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
superimposed_img=heatmap*0.4+img
cv2.imwrite('/Users/....jpg',superimposed_img)#保存至硬盘






































































