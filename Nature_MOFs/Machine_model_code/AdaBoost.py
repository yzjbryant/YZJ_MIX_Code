from sklearn.ensemble import GradientBoostingClassifier
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


model=GradientBoostingClassifier(n_estimators=100,
                                 learning_rate=0.1,
                                 max_depth=1,
                                 random_state=0)
model.fit(x_train,y_train)
predicted=model.predict(x_test)
print(predicted)