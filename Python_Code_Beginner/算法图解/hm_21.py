from numpy import *
e = eye(4)

print(e)

x = random.rand(4,4)
print(x) #4*4随机数组

#矩阵matrix 等价于Matlab中的matrices
#数组array
#mat()将数组转化为矩阵

randMat = mat(random.rand(4,4))
print(randMat)

#.I 操作符实现了矩阵求逆的运算



