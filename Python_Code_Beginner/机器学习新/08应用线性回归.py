# #In[1]:
# import pandas as pd
#
# dataframe = pd.read_csv('./winequality-red.csv',sep=';')
# dataframe.describe()
#
# import matplotlib.pylab as plt
#
# plt.scatter(dataframe['alcohol'],dataframe['quality'])
# plt.xlabel('Alcohol')
# plt.ylabel('Quality')
# plt.title('Alcohol Against Quality')
# plt.show()

#In[1]: 拟合和评估模型
# from sklearn.linear_model import LinearRegression
# import pandas as pd
# import matplotlib.pylab as plt
# from sklearn.model_selection import train_test_split
#
# dataframe = pd.read_csv('./winequality-red.csv',sep=';')
# X=dataframe[list(dataframe.columns)[:-1]]
# y=dataframe['quality']
# X_train,X_test,y_train,y_test=train_test_split(X,y)
# regressor=LinearRegression()
# regressor.fit(X_train,y_train)
# y_predictions=regressor.predict(X_test)
# print('R-squared:%s'%regressor.score(X_test,y_test))
#
# #In[2]:
# import pandas as pd
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LinearRegression
#
# df=pd.read_csv('./winequality-red.csv',sep=';')
# X=df[list(df.columns)[:-1]]
# y=df['quailty']
# regressor=LinearRegression()
# scores=cross_val_score(regressor,X,y,cv=5)
# print(scores.mean())
# print(scores)

#In[1]: 随机梯度下降法
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import  SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data=load_boston()
X_train,X_test,y_train,y_test=train_test_split(data.data,data.target)

X_scaler=StandardScaler()
y_scaler=StandardScaler()
X_train=X_scaler.fit_transform(X_train)
y_train=y_scaler.fit_transform(y_train.reshape(-1,1))
X_test=X_scaler.transform(X_test)
y_test=y_scaler.transform(y_test.reshape(-1,1))
regressor=SGDRegressor(loss='squared_loss')
scores=cross_val_score(regressor,X_train.y_train,cv=5)
print('Cross validation r-squared scores:%s'%scores)
print('Average cross validation r-squared score :%s'%np.mean(scores))
regressor.fit(X_train,y_train)
print('Test set r-squared score %s' % regressor.score(X_test,y_test))