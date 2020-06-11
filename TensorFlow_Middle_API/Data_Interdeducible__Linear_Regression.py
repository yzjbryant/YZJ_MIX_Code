import numpy as np

def linear_loss(X,y,w,b):
    num_train=X.shape[0]
    num_feature=X.shape[1]
    #模型公式
    y_hat=np.dot(X,w)+b
    #损失函数
    loss=np.sum((y_hat-y)**2)/num_train
    #参数的偏导
    dw=np.dot(X.T,(y_hat-y))/num_train
    db=np.sum((y_hat-y))/num_train
    return y_hat,loss,dw,db

#参数初始化
def initialize_params(dims):
    w=np.zeros((dims,1))
    b=0
    return w,b

#基于梯度下降的模型训练过程
def linear_train(X,y,learning_rate,epochs):
    w,b=initialize_params(X.shape[1])
    loss_list=[]
    for i in range(1,epochs):
        y_hat,loss,dw,db=linear_loss(X,y,w,b)
        loss_list.append(loss)
        w += learning_rate*dw
        b += learning_rate*db
