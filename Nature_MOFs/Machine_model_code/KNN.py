#通过打斗次数和接吻次数来界定电影类型
# import  numpy as np
# from sklearn import neighbors
#
# knn=neighbors.KNeighborsClassifier()
#
# data=np.array([[3,104],[2,100],[1,81],
#                [101,10],[99,5],[98,2]])
# labels=np.array([1,1,1,2,2,2])
# #1:Romance
# #2:Action
# knn.fit(data,labels)
# # print(knn.fit(data,labels))
# c=np.array([[16,17]])
# a=knn.predict(c)
# print(a)

import numpy as np
from sklearn import neighbors

data=np.array([[1,2],[3,4],[78,90],[1,90]])
label=np.array([1,2,1,1])

knn=neighbors.KNeighborsClassifier(n_neighbors=2)

knn.fit(data,label)
print(knn.predict([[1,3]]))












