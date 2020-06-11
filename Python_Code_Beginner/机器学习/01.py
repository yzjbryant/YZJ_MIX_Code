from numpy import *
e = eye(4)

print(e)

#4*4随机数组
x = random.rand(4,4)
print(x)

#矩阵matrix 等价于Matlab中的matrices
#数组array

#mat()将数组转化为矩阵
randMat = mat(random.rand(4,4))
print(randMat)

#.I 操作符实现了矩阵求逆的运算
invRandMat = randMat.I
print(invRandMat)

y = randMat * invRandMat
print(y)

#误差值
# r = y - e
r = randMat * invRandMat - eye(4)
print(r)



import tensorflow as tf

