##处理分类任务

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(0)
iris=datasets.load_iris()
x=iris.data
y=iris.target
i=np.random.permutation(len(iris.data))
x_train=x[i[:-10]]
y_train=y[i[:-10]]
x_test=x[i[-10:]]
y_test=y[i[-10:]]

knn=KNeighborsClassifier()
p=knn.fit(x_train,y_train)
print(p)
z=knn.predict(x_test)
print(z)
o=y_test
print(o)