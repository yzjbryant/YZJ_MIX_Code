# # In [1]:
# import numpy as np
# import matplotlib.pyplot as plt
# X=np.array([[6],[8],[10],[14],[18]]).reshape(-1,1)
# y=[7,9,13,17.5,18]
# plt.figure()
# plt.title('Pizza price plotted against diamater')
# plt.xlabel('Diamter in inches')
# plt.ylabel('Price in dollars')
# plt.plot(X,y,'k.')
# plt.axis([0,25,0,25])
# plt.grid(True)
# plt.show()
#
# # In [2]:
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(X,y)
# test_pizza=np.array([[8]])
# predicted_price=model.predict(test_pizza)[0]
# print('A 12" pizza should cost: $%.2f' % predicted_price)
# print('Residual sum of squares: %.2f' % np.mean((model.predict(X)-y)**2))
# #
#
# # #  In [2]:
# import numpy as np
# X=np.array([[6],[8],[10],[14],[18]]).reshape(-1,1)
# x_bar=X.mean()
# print(x_bar)
# variance=((X-x_bar)**2).sum()/(X.shape[0]-1)
# print(variance)
# print(np.var(X,ddof=1))
# #
# # # In [4]:
# y=np.array([7,9,13,17.5,18])
# y_bar=y.mean()
# covariance=np.multiply((X-x_bar).transpose(),y-y_bar).sum() / (X.shape[0]-1)
# print(covariance)
# print(np.cov(X.transpose(),y)[0][1])

# In [1]:
import numpy as np
from sklearn.linear_model import LinearRegression

X_train=np.array([6,8,10,14,18]).reshape(-1,1)
y_train=[7,9,13,17.5,18]

X_test=np.array([8,9,11,16,12]).reshape(-1,1)
y_test=[11,8.5,15,18,11]

model=LinearRegression()
model.fit(X_train,y_train)
r_squared=model.score(X_test,y_test)
print(r_squared)