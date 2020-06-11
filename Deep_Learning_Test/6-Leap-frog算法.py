#混合蛙跳算法原理的简单实现
#问题描述：现有若干任务，并有若干台处理任务的机器，每台
#机器处理任务的速度不同，找出最好的安排方案

import matplotlib.pyplot as plt
import numpy as np
import random
import operator

#适应度函数
def F(plan):
    sum=[]
    for i in range(d):
        sum.append(0)
    for i in range(d):
        if (plan[i]<0|plan[i]>nodes.__len__()):
            return 100
        sum[plan[i]] += round(task[i]/nodes[plan[i]],3)
        #任务学习的步长最大为3，因为只有三个节点
    sum.sort(reverse=True)
    return sum[0]

#初始化
task=[1,2,3,4,5,6,7,8,9] #任务长度
nodes=[1,2,3] #节点

#SFLA参数
N=100 #种群数，有100种分配方案
n=10 #子群中的青蛙数量，10个分配方案为一组
d=task.__len__()
m=N//n #子群数量，共10组
L=5 #组内优化更新最差青蛙次数，组内优化的最差解的步数
G=100 #种群迭代次数
D_max=10
P=[]

#step1 生成蛙群，生成由0~2组成有9个元素的数组
for i in range(N):
    t=[[],0]  #t[[解],适应值]
    for j in range(d):
        t[0].append(random.randint(0,2))
    t[0]=np.array(t[0])
    t[1]=F(t[0])
    print(t[1])
    P.append(t)
P.sort(key=operator.itemgetter(1))
Xg=P[0]  #首个全局最优解

for k in range(G):
    #step2 划分子群
    M=[]
    for i in range(m):
        M.append([]) #10个空的族群
    for i in range(N):
        M[i%m].append(P[i]) #i%10 0~9 P长度为100
    Xb=[]
    Xw=[]
    for i in range(m):
        Xb.append(M[i][0])
        Xw.append(M[i][M[i].__len__()-1])

#step3 局部搜索
for i in range(m):
    for j in range(L):
        D=random.randint(0,1)*(Xb[i][0]-Xw[i][0])
        #Xb[i][0]是解空间
        temp=Xw[i][0]+D
        if (F(temp)<F(Xw[i][0])):
            f=0
            Xw[i][0]=temp
            Xw[i][1]=F(temp) #最差解被替换
            M[i][M[i].__len__()-1]=Xw[i] #i族群中最后一个是最差解，用新解替换
        else:
            Xb[i]=Xg #适应值没有变小，用全局最优替换局部最优
            f=2
        if (f==2):
            t=[[],0]
            for j in range(d):
                t[0].append(random.randint(0,2))
            t[0]=np.array(t[0])
            t[1]=F(t[0])
            Xw[i]=t
    P=[]
    for i in range(m):
        for j in range(n):
            P.append(M[i][j])

    #sorted(P,key=lambda P:P[1])
    P.sort(key=operator.itemgetter(1))
    Xg=P[0]
    x=[]
    y=[]
    for i in range(P.__len__()):
        x.append(k)
        y.append(P[i][1])
    plt.scatter(x,y,s=5)
print(P[0])
plt.show()

















































