import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data=[]
label=[]

with open('./data.txt') as f:
    for line in f:
        token=line.strip().split(' ')
        data.append([float(tk) for tk in token[:-1]])
        label.append(token[-1])
x=np.array(data)
label=np.array(label)
y=np.zeros(label.shape)

y[label=='fat']=1


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

model=GaussianNB()
model.fit(x,y)
predicted=model.predict(x_test)
print(x_test)
print(predicted)
