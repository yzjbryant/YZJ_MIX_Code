from sklearn import svm
svc=svm.SVC(gamma=0.001,C=100.)
from sklearn import datasets
digits=datasets.load_digits()
print(digits.DESCR)#说明文档

print(digits.images[0])
import matplotlib.pyplot as plt
plt.imshow(digits.images[0],
           cmap=plt.cm.gray_r,
           interpolation='nearest')
# plt.show()
print(digits.target)
print(digits.target.size)

a=svc.fit(digits.data[1:1790],digits.target[1:1790])
print(a)

b=svc.predict(digits.data[1791:1976])
print(b)
