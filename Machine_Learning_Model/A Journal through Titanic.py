import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('./titanic/train.csv',header=0)
data_test=pd.read_csv('./titanic/test.csv',header=0)

#整体信息
print(data.info())
print(data_test.info())
print("==============================")

#前五行
print(data.head())
print(data_test.head())
print("==============================")

#统计信息描述
print(data.describe())
print(data_test.describe())
print("==============================")

#查看变量类型
print(data.dtypes)
print(data_test.dtypes)
print("==============================")


#########处理各字段信息
#1.drop unnecessary columns
data=data.drop(['PassengerId','Name','Ticket'],axis=1)
data_test=data_test.drop(['Name','Ticket'],axis=1)

#2.处理Sex字段
data['Gender']=data['Sex'].map({'female':0,'male':1}).astype(int)
data_test['Gender']=data_test['Sex'].map({'female':0,'male':1}).astype(int)
print(data['Gender'].head())
print(data_test['Gender'].head())

#3.处理Age字段
print(data['Age'].mean())
print(data['Age'].median())
#3.1筛选数据
print(data[data['Age']>60])
print(data[data['Age']>60][['Sex','Pclass','Age','Survived']])
print(data[data['Age'].isnull()][['Sex','Pclass','Age']])
#3.2绘制直方图
data['Age'].dropna().hist(bins=16,range=(0,80),alpha=0.5)
# plt.show()

#根据性别和客舱级别求年龄的平均值
median_ages=np.zeros((2,3))
median_ages_test=np.zeros((2,3))
print(median_ages)
print(median_ages_test)

for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j]=data[(data['Gender']==i)&(data['Pclass']==j+1)]['Age'].dropna().median()
        median_ages_test[i,j] = data_test[(data_test['Gender'] == i) & (data_test['Pclass'] == j + 1)]['Age'].dropna().median()
print(median_ages)
print(median_ages_test)
data['AgeFill']=data['Age']
data_test['AgeFill']=data_test['Age']
print(data.head())
print(data[data['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(10))

#填充缺失值
for i in range(0,2):
    for j in range(0,3):
        data.loc[(data.Age.isnull())&(data.Gender==i)&(data.Pclass==j+1),'AgeFill']=median_ages[i,j]
        data_test.loc[(data_test.Age.isnull()) & (data_test.Gender == i) & (data_test.Pclass == j + 1), 'AgeFill'] = median_ages_test[i, j]

print(data[data['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(10))
print(data_test[data_test['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(10))

#创建字段，记录Age中哪些是原有的，哪些是后加入的
data['AgeIsNull']=pd.isnull(data.Age).astype(int)
data_test['AgeIsNull']=pd.isnull(data_test.Age).astype(int)
print(data['AgeIsNull'].head(10))
print(data_test['AgeIsNull'].head(10))

#(4)Using sex and age for person column
def get_person(passenger):
    age,sex=passenger
    return 'child' if age<16 else sex
data['Person']=data[['Age','Sex']].apply(get_person,axis=1)
data_test['Person']=data_test[['Age','Sex']].apply(get_person,axis=1)

person_dummies=pd.get_dummies(data['Person'])
person_dummies.columns=['Child','Female','Male']
person_dummies.drop(['Male'],axis=1,inplace=True)

person_dummies_test=pd.get_dummies(data_test['Person'])
person_dummies_test.columns=['Child','Female','Male']
person_dummies_test.drop(['Male'],axis=1,inplace=True)

data=data.join(person_dummies)
data_test=data_test.join(person_dummies_test)

#(5)处理Embarked字段
data['Embarked'].describe()
data['Embarked'].value_counts()
temp1=data["Embarked"].value_counts()
temp1.plot(kind='bar')
# plt.show()
temp2=data.pivot_table(values="Survived",
                       index=["Embarked"],
                       aggfunc=lambda x:x.mean())
temp2.plot(kind='bar')
# plt.show()

embark_dummies=pd.get_dummies(data['Embarked'])
embark_dummies.drop(['S'],axis=1,inplace=True)
embark_dummies_test=pd.get_dummies(data_test['Embarked'])
embark_dummies_test.drop(['S'],axis=1,inplace=True)

#原数据集与哑变量合并
data=data.join(embark_dummies)
data_test=data_test.join(embark_dummies_test)
print(data.head(),data_test.head())

#(6)处理Fare字段
#标准化处理
data["Fare"].fillna(data["Fare"].median(),inplace=True)
Fare_avg=data["Fare"].mean()
Fare_std=data["Fare"].std()
data["Fare"]=(data["Fare"]-Fare_avg)/Fare_std
data["Fare"].hist(bins=40)

data_test["Fare"].fillna(data_test["Fare"].median(),inplace=True)
Fare_avg_test=data_test["Fare"].mean()
Fare_std_test=data_test["Fare"].std()
data_test["Fare"]=(data_test["Fare"]-Fare_avg_test)/Fare_std_test

#(7)处理Family字段
data['FamilySize']=data['SibSp']+data['Parch']
data_test['FamilySize']=data_test['SibSp']+data_test['Parch']
data['FamilySize'].loc[data['FamilySize']>0]=1
data['FamilySize'].loc[data['FamilySize']==0]=0
print(data['FamilySize'].value_counts())


####定义训练集和测试集
X_train=data.drop("Survived",axis=1)
Y_train=data["Survived"]
X_test=data_test.drop("PassengerId",axis=1).copy()

##5.建模分析
#machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#Logistic Regression
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred=logreg.predict(X_test)
logreg.score(X_train,Y_train)
##0.8002244

#SVM
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred=svc.predict(X_test)
svc.score(X_train,Y_train)
##0.81

#Linear SVC
svc=LinearSVC()
svc=svc.fit(X_train,Y_train)
Y_pred=svc.predict(X_test)
svc.score(X_train,Y_train)
##0.8035

#Random Forest
random_forest=RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,Y_train)
Y_pred=random_forest.predict(X_test)
random_forest.score(X_train,Y_train)
#0.92

#KNN
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
Y_pred=knn.predict(X_test)
knn.score(X_train,Y_train)
#0.868

#Gaussian Naive Bayes
gaussian=GaussianNB()
gaussian.fit(X_train,Y_train)
Y_pred=gaussian.predict(X_test)
gaussian.score(X_train,Y_train)

##correlation analysis
coeff_df=DataFrame(data.columns.delete(0))
coeff_df.columns=['Features']
coeff_df["Coefficient Estimate"]=pd.Series(logreg.coef_[0])
coeff_df
submission=pd.DataFrame({"PassengerId":data_test["PassengerId"],
                         "Survived":Y_pred})
submission.to_csv('titanic.csv',index=False)