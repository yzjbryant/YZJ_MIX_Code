# import math
# import matplotlib.pyplot as plt
import numpy as np

# if __name__=="__main__":
#     x=np.array(np.linspace(start=-3,stop=3,num=1001,dtype=np.float))
#     y_logit=np.log(1+np.exp(-x))/math.log(2)
#     y_01=x<0
#     y_hinge=1.0-x
#     y_hinge[y_hinge<0]=0
#     plt.plot(x,y_logit,'r--',label='Logistic Loss',linewidth=2)
#     plt.plot(x,y_01,'g-',label='0/1 Loss',linewidth=2)
#     plt.plot(x,y_hinge,'b-',label='Hinge LOss',linewidth=2)
#     plt.grid()
#     plt.legend(loc='upper right')
#     # plt.savefig('1.png')
#     plt.show()


# data=np.random.random((3,2))
# print(data)
# # print(data[0:2])
# # print(data.T.max(axis=0))
# print(data.reshape(2,3))
#
# dara=np.array([[1,2],[3,4],
#                [5,6],[7,8]])
# print(dara)
#
# dadd=np.ones((4,3,2))
# print(dadd)

# n=2
# predictions=np.zeros((3,2))
# labels=np.ones((3,2))
# error=1/n*np.sum(np.square(predictions-labels))
# print(error)

# import pandas as pd
# data=pd.read_excel("co2.xlsx")
# print(data)

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score,fbeta_score
from sklearn.metrics import precision_recall_fscore_support
if __name__=="__main__":
    y_ture=np.array([1,1,1,1,0,0])
    y_hat=np.array([1,0,1,1,1,1])
    print('Accuracy:\t', accuracy_score(y_ture,y_hat))