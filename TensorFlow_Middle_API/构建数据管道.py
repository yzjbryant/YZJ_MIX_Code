# 1. 从Numpy array构建数据管道

import tensorflow as tf
import numpy as np
from sklearn import datasets

iris=datasets.load_iris()

dsl=tf.data.Dataset.from_tensor_slices((iris["data"],iris["target"]))

for features, label in dsl.take(5):
    print(features,label)


# 2. 从Pandas DataFrame构建数据管道

import tensorflow as tf
from sklearn import datasets
import pandas as pd

iris=datasets.load_iris()

dfiris=pd.DataFrame(iris["data"],columns=iris.feature_names)

ds2=tf.data.Dataset.from_tensor_slices((dfiris.to_dict("list"),iris["target"]))

for features,label in ds2.take(3):
    print(features,label)

# 3. 从Python generator 构建数据管道

import tensorflow as tf
from matplotlib import pyplot as plt
from  tensorflow.keras.preprocessing.image import ImageDataGenerator









