##生成器
#GAN生成器网络
import keras
from keras import layers
import numpy as np

latent_dim=32
height=32
width=32
channels=3

generator_input=keras.Input(shape=(latent_dim,))

x=layers.Dense(128*16*16)(generator_input)
x=layers.LeakyReLU()(x)
x=layers.Reshape((16,16,128))(x)

x=layers.Conv2D(256,5,padding='same')(x)
x=layers.LeakyReLU()(x)

x=layers.Conv2DTranspose(256,4,strides=2,padding='same')(x)
x=layers.LeakyReLU()(x)

x=layers.Conv2D(256,5,padding='same')(x)
x=layers.LeakyReLU()(x)
x=layers.Conv2D(256,5,padding='same')(x)
x=layers.LeakyReLU()(x)

x=layers.Conv2D(channels,7,activation='tanh',padding='same')(x)
generator=keras.models.Model(generator_input,x)
generator.summary()


#判别器
discriminator_input=layers.Input(shape=(height,width,channels))
x=layers.Conv2D(128,3)(discriminator_input)
x=layers.LeakyReLU()(x)
x=layers.Conv2D(128,4,strides=2)(x)
x=layers.LeakyReLU()(x)
x=layers.Conv2D(128,4,strides=2)(x)
x=layers.LeakyReLU()(x)
x=layers.Conv2D(128,4,strides=2)(x)
x=layers.LeakyReLU()(x)
x=layers.Flatten()(x)

x=layers.Dropout(0.4)(x)

x=layers.Dense(1,activation='sigmoid')(x)

discriminator=keras.models.Model(discriminator_input)
discriminator.summary()

discriminator_optimizer=keras.optimizers.RMSprop(lr=0.0008,clipvalue=1.0,decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy')

###对抗网络

discriminator.trainable=False

gan_input=keras.Input(shape=(latent_dim,))
gan_output=discriminator(generator(gan_input))
gan=keras.models.Model(gan_input,gan_output)
gan_optimizer=keras.optimizers.RMSprop(lr=0.0004,clipvalue=1.0,decay=1e-8)
gan.compile(optimizer=gan_optimizer,loss='binary_crossentropy')


#实现GAN训练
import os
from keras.preprocessing import image

(x_train,y_train),(_,_)=keras.datasets.cifar10.load_data()
x_train=x_train[y_train.flatten()==6]
x_train=x_train.reshape((x_train.shape[0],)+(height,width,channels)).astype('float32')/255.

iteractions=10000
batch_size=20
save_dir='your_dir'


























