import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt

#设置模型参数
params={
    'booster':'gbtree',
    'objective':'multi:softmax',
    'num_class':3,
    'gamma':0.1,
    'max_depth':2,
    'lambda':2,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':3,
    'silent':1,
    'eta':0.001,
    'seed':1000,
    'nthread':4,
}
plst=params.items()
print(plst)

dtrain=xgb.DMatrix(X_train,y_train)
num_rounds=200
model=xgb.train(plst,dtrain,num_rounds)

#对测试集进行预测
dtest=xgb.DMatrix(X_test)
y_pred=model.predict(dtest)
#计算准确率
accuracy=accuracy_score(y_test,y_pred)
plot_importance(model)
plt.show()















