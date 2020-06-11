# In [2]:
import numpy as np
import matplotlib.pyplot as plt

X_train=np.array([
    [158,64],
    [170,86],
    [183,84],
    [191,80],
    [155,49],
    [180,67],
    [158,54],
    [170,67]
])
y_train=['male','male','male','male','female','female','female','female','female']
plt.figure()
plt.title('Human Heights and Weights by Sex')
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')

for i,x in enumerate(X_train):
    plt.scatter(x[0],x[1],c='k',marker='x' if y_train[i] == 'male' else 'D')

plt.grid(True)
plt.show()

# In [2]:
x=np.array([155,70])
distances=np.sqrt(np.sum((X_train-x)**2, axis=1))
print(distances)

#In [3]:
nearest_neighbor_indices=distances.argsort()[:3]
nearest_neighbor_genders=np.take(y_train, nearest_neighbor_indices)
print(nearest_neighbor_genders)

# In [4]:
from collections import Counter
b=Counter(np.take(y_train,distances.argsort()[:3]))
print(b.most_common(1)[0][0])

#In [5]:
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier

lb=LabelBinarizer()
y_train_binarized=lb.fit_transform(y_train)
print(y_train_binarized)

#In [6]:
K=3
clf=KNeighborsClassifier(n_neighbors=K)
clf.fit(X_train,y_train_binarized.reshape(-1))
prediction_binarized=clf.predict(np.array([155,70]).reshape(1,-1))[0]
predicted_label=lb.inverse_transform((prediction_binarized))
print(predicted_label)

#In [7]:
X_test=np.array(
    [168,65],
    [180,96],
    [160,52],
    [169,67]
)
y_test=['male','male','female','female']
y_test_binarized=lb.transform(y_test)
print('Binarized labels: %s' % y_test_binarized.T[0])
prediction_binarized=clf.predict(X_test)
print('Binarized predictions: %s' % prediction_binarized)
print('Predicted labels: %s' % lb.inverse_transform(prediction_binarized))

#In [8]: 准确率
from sklearn.metrics import accuracy_score
print('Accuracy:%s'%accuracy_score(y_test_binarized,prediction_binarized))

#In [9]: 精确率
from sklearn.metrics import precision_score
print('Precision :  %s' % precision_score(y_test_binarized,prediction_binarized))

#In [10]: 召回率
from sklearn.metrics import recall_score
print('Recall:%s'%recall_score(y_test_binarized,prediction_binarized))

# In[11]: F1 得分是精确率和召回率的调和平均值
from sklearn.metrics import f1_score
print('F1 score:%s '% f1_score(y_test_binarized,prediction_binarized))

# In[12]: 马修斯相关系数
from sklearn.metrics import matthews_corrcoef
print('Matthews correlation coefficient : %s' % matthews_corrcoef(y_test_binarized,prediction_binarized))

# In[13]:
from sklearn.metrics import classification_report
print(classification_report(y_test_binarized,prediction_binarized,
                            target_names=['male'],labels=[1]))

