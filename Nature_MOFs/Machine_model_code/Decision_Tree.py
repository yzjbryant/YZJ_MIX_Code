# import math
# #Python实现熵的计算
# def calcShannonEnt(dataSet):
#     numEntries=len(dataSet)
#     labelCount={}
#     for featVec in dataSet:
#         currentLabel=featVec[-1]
#         if currentLabel not in labelCount.keys():
#             labelCount[currentLabel]=0
#         labelCount[currentLabel]+=1
#     shannonEnt=0.0
#     for key in labelCount:
#         prob=float(labelCount[key])/numEntries
#         shannonEnt-=prob*math.log(prob,2)
#     return shannonEnt
# c=['a','b','c','d','e','1']
# a=calcShannonEnt(c)
# print(a)

#实战
#-*-coding:utf-8-*-
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

#使用信息熵进行训练
clf=tree.DecisionTreeClassifier(criterion='entropy')

clf.fit(x_train,y_train)

with open("tree.dot",'w') as f:
    f=tree.export_graphviz(clf,out_file=f)

# print(clf.feature_importances_)

# answer=clf.predict(x_train)
# print(x_train)
# print(answer)
# print(y_train)
# print(np.mean(answer==y_train))
#
# precision,recall,thresholds=precision_recall_curve(y_train,clf.predict(x_train))
# answer=clf.predict_log_proba(x)[:,1]
# print(classification_report(y,answer,target_names=['thin','fat']))


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=2000)
model.fit(x_train,y_train)
predicted=model.predict(x_test)
print(predicted)











