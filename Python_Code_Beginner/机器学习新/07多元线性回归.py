#In[1]:
# from numpy.linalg import inv
# from numpy import dot, transpose
#
# X=[[1,6,2],[1,8,1],[1,10,0],[1,14,2],[1,18,0],]
# y=[[7],[9],[13],[17.5],[18]]
# # print(dot(inv(dot(transpose(X),X)),dot(transpose(X),y)))
#
# #In[2]: 最小二乘函数
# from numpy.linalg import lstsq
# print(lstsq(X,y),[0])

#In[3]:使用第二个解释变量来更新披萨价格
# from sklearn.linear_model import LinearRegression
# X=[[6,2],[8,1],[10,0],[14,2],[18,0]]
# y=[[7],[9],[13],[17.5],[18]]
# model=LinearRegression()
# model.fit(X,y)
# X_test=[[8,2],[9,0],[11,2],[16,2],[12,0]]
# y_test=[[11],[8.5],[15],[18],[11]]
# predictions=model.predict(X_test)
# for i,prediction in enumerate(predictions):#枚举
#     print('Predicted:%s, Target:%s'%(prediction,y_test[i]))
#     print('R-squared:%.2f'%model.score(X_test,y_test))


#In[4]: 多项式回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X_train=[[6],[8],[10],[14],[18]]
y_train=[[7],[9],[13],[17.5],[18]]
X_test=[[6],[8],[10],[14]]
y_test=[[7],[9],[13],[17.5]]
regressor=LinearRegression()
regressor.fit(X_train,y_train)
xx=np.linspace(0,26,100)
yy=regressor.predict(xx.reshape(xx.shape[0],1))
plt.plot(xx,yy)
# print(yy)
# print(xx)

quadratic_featurizer=PolynomialFeatures(degree=9)
X_train_quadratic=quadratic_featurizer.fit_transform(X_train)
# print(X_train_quadratic)
X_test_quadratic=quadratic_featurizer.transform(X_test)
# print(X_test_quadratic)
regressor_quadratic=LinearRegression()
regressor_quadratic.fit(X_train_quadratic,y_train)
xx_quadratic=quadratic_featurizer.transform(xx.reshape(xx.shape[0],1))
plt.plot(xx,regressor_quadratic.predict(xx_quadratic),c='r',linestyle='--')
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter')
plt.ylabel('Price in dollars')
plt.axis([0,25,0,25])
plt.grid(True)
plt.scatter(X_train,y_train)
plt.show()

print(X_train)
print(X_train_quadratic)
print(X_test)
print(X_test_quadratic)
print('Simple linear regression r-squared',regressor.score(X_test,y_test))
print('Quadratic regression r-squared',
      regressor_quadratic.score(X_test_quadratic,y_test))




