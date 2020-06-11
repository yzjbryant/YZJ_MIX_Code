from sklearn import datasets
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

diabetes=datasets.load_diabetes()
linreg=linear_model.LinearRegression()

x_train=diabetes.data[:-20]
y_train=diabetes.target[:-20]
x_test=diabetes.data[-20:]
y_test=diabetes.target[-20:]

plt.figure(figsize=(8,12))
for f in range(0,10):
    xi_test=x_test[:,f]
    xi_train=x_train[:,f]
    xi_test=xi_test[:,np.newaxis]
    xi_train=xi_train[:,np.newaxis]
    linreg.fit(xi_train,y_train)
    y=linreg.predict(xi_test)
    plt.subplot(5,2,f+1)
    plt.scatter(xi_test,y_test,color='k')
    plt.plot(xi_test,y,color='b',linewidth=3)
    plt.show()