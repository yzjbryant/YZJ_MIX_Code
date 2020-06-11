##二元或判别模型
#SVR-支持向量回归
#SVC-支持向量分类

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

x=np.array([[1,3],[1,2],[1,1.5],[1.5,2],[2,3],
           [2.5,1.5],[2,1],[3,1],[3,2],[3.5,1],[3.5,3]])
y=[0]*6+[1]*5

# plt.scatter(x[:,0],x[:,1],c=y,s=50,alpha=0.9)
# plt.show()

svc=svm.SVC(kernel='linear').fit(x,y)
X,Y=np.mgrid[0:4:200j,0:4:200j]
Z=svc.decision_function(np.c_[X.ravel(),Y.ravel()])
Z=Z.reshape(X.shape)
plt.contourf(X,Y,Z>0,alpha=0.4)
plt.contourf(X,Y,Z,colors=['k'],linestyles=['-'],levels=[0])
plt.scatter(x[:,0],x[:,1],c=y,s=50,alpha=0.9)
plt.show()