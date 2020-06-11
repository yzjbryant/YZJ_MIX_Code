from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


iris=datasets.load_iris()
print(iris.data.shape)##行与列
print(iris.target)##iris的属性
print(iris.target_names)##iris类别


x=iris.data[:,0]  #X-Axis-sepal length
y=iris.data[:,1]  #Y-Axis-sepal length

# x=iris.data[:,2]  #X-Axis-petal length
# y=iris.data[:,3]  #Y-Axis-petal length

species=iris.target #Species

x_min,x_max=x.min()-.5,x.max()+.5
y_min,y_max=y.min()-.5,y.max()+.5

#Scatterplot
plt.figure()
plt.title('Iris Dataset - Classification By Sepal Sizes')

# plt.title('Iris Dataset - Classification By Petal Sizes')

plt.scatter(x,y,c=species)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# plt.xlabel('Petal length')
# plt.ylabel('Petal width')

plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xticks(())
plt.yticks(())
plt.show()


##主成分分解
#fit_transfrom()用来降维，属于PCA对象
#要导入sklearn.decomposition

from sklearn.decomposition import PCA
x_reduced=PCA(n_components=3).fit_transform(iris.data)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA





















