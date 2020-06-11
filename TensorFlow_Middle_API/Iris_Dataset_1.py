##主成分分解
#fit_transfrom()用来降维，属于PCA对象
#要导入sklearn.decomposition


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA


iris=datasets.load_iris()
x=iris.data[:,1]
y=iris.data[:,2]
species=iris.target
x_reduced=PCA(n_components=3).fit_transform(iris.data)

#ScatterPlot 3D
fig=plt.figure()
ax=Axes3D(fig)
ax.set_title('Iris Dataset by PCA',size=14)
ax.scatter(x_reduced[:,0],x_reduced[:,1],x_reduced[:,2],c=species)
ax.set_xlabel('First eigenvector')
ax.set_ylabel('Second eigenvector')
ax.set_zlabel('Third eigenvector')
ax.w_xaxis.set_ticklabels(())
ax.w_yaxis.set_ticklabels(())
ax.w_zaxis.set_ticklabels(())
plt.show()