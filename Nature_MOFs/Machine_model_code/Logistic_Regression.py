#辨别不同的因素对研究生录取的影响

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pylab as pl

df=pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")
print(df.head())

df.columns=["admit","gre","gpa","prestige"]
print(df.columns)
print(df.describe())
print(df.std())
print(pd.crosstab(df['admit'],df['prestige'],rownames=['admit']))
print(df.hist())
pl.show()
dummy_ranks=pd.get_dummies(df['prestige'],
                           prefix='prestige')
print(dummy_ranks.head())
cols_to_keep=['admit','gre','gpa']
data=df[cols_to_keep].join(dummy_ranks.ix[:,'prestige_2':])
print(data.head())
data['intercept']=1

#执行逻辑回归
#预测admit列
train_cols=data.columns[1:]
logit=sm.Logit(data['admit'],data[train_cols])
result=logit.fit()

#预测数据
import copy
combos=copy.deepcopy(data)
predict_cols=combos.columns[1:]
combos['intercept']=1
combos['predict']=result.predict(combos[predict_cols])

total=0
hit=0
for value in combos.values:
    predict=value[-1]
    admit=int(value[0])
    if predict>0.5:
        total+=1
        if admit==1:
            hit+=1

print(result.summary())
print(result.conf_int())
print(np.exp(result.params))
