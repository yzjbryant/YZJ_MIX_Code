import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
a=np.arange(0,60,10).reshape((-1,1))+np.arange(6)
print(a)
print(a[(0,1,2,3),(2,3,4,5)])
print(a[3:,[0,2,5]])
i=np.array([True,False,True,False,False,True])
print(a[i])
print(a[i,3])


mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False
mu=0
sigma=1
x=np.linspace(mu-3*sigma, mu+3*sigma,50)
y=np.exp(-(x-mu)**2/(2*sigma**2))/(math.sqrt(2*math.pi)*sigma)
print(x.shape)
print('x=\n',x)
print(y.shape)
print('y=\n',y)
plt.plot(x,y,'r-',x,y,'go',linewidth=2,markersize=8)
plt.grid(True)
plt.title("高斯分布")
plt.show()

#损失函数：logistic(-1,1)/SVM Hinge损失/0/1损失
x=np.array(np.linspace(start=-2,stop=3,num=1001,dtype=np.float))
y_logit=np.log(1+np.exp(-x))/math.log(2)
y_boost=np.exp(-x)
y_01=x<0
y_hinge=1.0-x
y_hinge[y_hinge<0]=0
plt.plot(x,y_logit,'r-',label='Logistic Loss',linewidth=2)
plt.plot(x,y_01,'g-',label='0/1 Loss',linewidth=2)
plt.plot(x,y_hinge,'b-',label='Hinge Loss',linewidth=2)
plt.plot(x,y_boost,'m--',label='Adaboost Loss',linewidth=2)
plt.grid()
plt.legend(loc='upper right')
plt.show()
#plt.savefig('1.png')

#x**x x>0
#(-x)**(-x) x<0
#
# def f(x):
#     y=np.empty_like(x)
#     i=x>0
#     y[i]=np.power(x[i],x[i])
#     i=x<0
#     y[i]=np.power(-x[i],-x[i])
#     # i=(x==0)
#     # y[i]=1
#     return y
#
# if __name__=="__main__":
mu=2
sigma=3
data=mu+sigma*np.random.randn(1000)
print(data)
h=plt.hist(data,30,normed=1,color='#a0a0ff')
x=h[1]
y=norm.pdf(x,loc=mu,scale=sigma)
plt