from sklearn.cluster import KMeans
import numpy as np
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

data=[]
label=[]
with open('./data.txt') as ifile:
    for line in ifile:
        tokens=line.strip().split(' ')
        data.append([float(tk) for tk in tokens[:-1]])
        label.append(tokens[-1])
x=np.array(data)
label=np.array(label)
y=np.zeros(label.shape)

#将标签转化为0，1
y[label=='fat']=1

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

clf=KMeans(n_clusters=3,max_iter=300,n_init=10)
clf.fit(x_train)
ypred=clf.predict(x_test)
