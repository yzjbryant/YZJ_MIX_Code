import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostClassifier,AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC,LinearSVR,SVR,NuSVC,NuSVR,OneClassSVM
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn import preprocessing
# import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import Ridge,Lasso,ElasticNet,LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#导入数据
data=pd.read_excel('CIF_Parameters.xlsx',header=0)
pd.set_option('display.max_rows', 6000)
pd.set_option('display.max_columns', 6000)
pd.set_option('display.width', 1000)

# 
# print(data)
# print("data============================================")
# print(data.info())
# print("data.info()============================================")
# print(data.head())
# print("data.head()============================================")
# print(data.describe())
# print("data.describe()============================================")
# print(data.dtypes)
# print("data.dtypes============================================")
# 处理各字段
# X_train=data.drop(["Materials"],axis=1)
# Y_train=data["Energy_C4H4S_298K_10kpa"]
# X_test=data_test.drop(["Materials","Energy_C4H4S_298K_10kpa"],axis=1).copy()
# 


#定义训练集与测试集

#定义测试集与训练集

# Energy_C4H4S_298K_10kpa	Energy_C4H4S_363K_10kpa	Energy_C4H4S_363K_100kpa
# Energy_C6H6_298K_10kpa	Energy_C6H6_363K_10kpa	Energy_C6H6_363K_100kpa
# Loading_C4H4S_298K_10kpa	Loading_C4H4S_363K_10kpa Loading_C4H4S_363K_100kpa
# Loading_C6H6_298K_10kpa	Loading_C6H6_363K_10kpa	Loading_C6H6_363K_100kpa

Target=['Loading_C4H4S_298K_10kpa','Loading_C4H4S_363K_10kpa','Loading_C4H4S_363K_100kpa',
        'Loading_C6H6_298K_10kpa','Loading_C6H6_363K_10kpa','Loading_C6H6_363K_100kpa']

X=data.drop(["Materials","Length_a","Length_b","Length_c","Number of Element",
             "Energy_C4H4S_298K_10kpa","Energy_C4H4S_363K_10kpa","Energy_C4H4S_363K_100kpa",
             "Energy_C6H6_298K_10kpa","Energy_C6H6_363K_10kpa","Energy_C6H6_363K_100kpa"],axis=1)
Y=data["Loading_C4H4S_298K_10kpa"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
# print(X_train,X_test,Y_train,Y_test)

#数据预处理
#标准化
scaler=StandardScaler(copy=True,with_mean=True,with_std=True).fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

'''
#归一化  无用
# scaler=Normalizer().fit(X_train)
# X_train=scaler.transform(X_train)
# X_test=scaler.transform(X_test)
'''

#############################################################################################建模分析
#file:///D:/thesis/Python/Python书籍/书籍/1/AI%20算法工程师手册/工具/scikit-learn/3.supervised_model.html

'''
##################################################################################Logistic Regression
#一、线性模型
#1.1 LinearRegression-线性回归模型
linear=LinearRegression(fit_intercept=True,normalize=False,
            copy_X=True,n_jobs=1)
linear.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=linear.predict(X_test.astype('int'))
a=linear.score(X_train.astype('int'),Y_train.astype('int'))
print("LinearRegression:",a)



#1.2 Ridge-岭回归
ridge=Ridge(alpha=1.0,fit_intercept=True,normalize=False,
            copy_X=True,max_iter=None,tol=0.001,
            solver='auto',random_state=None)
ridge.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=ridge.predict(X_test.astype('int'))
a=ridge.score(X_train.astype('int'),Y_train.astype('int'))
print("Ridge:",a)

#1.3 Lasso-Lasso回归
lasso=Lasso(alpha=1.0,fit_intercept=True,normalize=False,
            copy_X=True,max_iter=1000,tol=0.0001,warm_start=False,
            positive=False,random_state=None,selection='cyclic')
            #Option of selection: random / cyclic
lasso.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=lasso.predict(X_test.astype('int'))
a=lasso.score(X_train.astype('int'),Y_train.astype('int'))
print("Lasso:",a)

#1.4 ElasticNet-ElasticNet回归
elasticnet=ElasticNet(alpha=1.0,l1_ratio=0.5,fit_intercept=True,
                      normalize=False,copy_X=True,max_iter=1000,
                      tol=0.0001,warm_start=False,precompute=False,
                      positive=False,random_state=None,
                      selection='cyclic')
                      #Option of selection: random / cyclic
elasticnet.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=elasticnet.predict(X_test.astype('int'))
a=elasticnet.score(X_train.astype('int'),Y_train.astype('int'))
print("ElasticNet:",a)

#1.5 LogisticRegression-对数几率回归模型
logreg=LogisticRegression(penalty='l2',dual=False,tol=0.0001,
                          C=1.0,fit_intercept=True,intercept_scaling=1,
                          class_weight=None,random_state=None,
                          solver='liblinear',max_iter=100,multi_class='ovr',
                          verbose=0,warm_start=False,n_jobs=1)
                          #option of solver : newton-cg, lbfgs, liblinear, sag
logreg.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=logreg.predict(X_test.astype('int'))
a=logreg.score(X_train.astype('int'),Y_train.astype('int'))
b=logreg.predict_log_proba(X_train)
c=logreg.predict_proba(X_train)
print("Logistic:",a,b,c)

#1.6 LinearDiscriminantAnalysis-线性判别分析模型
lineardisana=LinearDiscriminantAnalysis(solver='svd',shrinkage=None,
                                        priors=None,n_components=None,
                                        store_covariance=False,tol=0.0001)
                    #option of solver : svd,lsqr,eigen
lineardisana.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=lineardisana.predict(X_test.astype('int'))
a=lineardisana.score(X_train.astype('int'),Y_train.astype('int'))
b=lineardisana.predict_log_proba(X_train)
c=lineardisana.predict_proba(X_train)
print("LinearDiscrimiantAnalysis:",a,b,c)
'''

'''
#################################################################################LinearSVC
#二、支持向量机
#2.1 LinearSVC
svc=LinearSVC(penalty='l2',loss='squared_hinge',dual=True,
              tol=0.0001,C=1.0,multi_class='ovr',
              fit_intercept=True,intercept_scaling=1,
              class_weight=None,verbose=0,random_state=None,
              max_iter=1000)
svc.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=svc.predict(X_test.astype('int'))
a=svc.score(X_train.astype('int'),Y_train.astype('int'))
print("LinearSVC:",a)

#2.2 SVC
svc=SVC(C=1.0,kernel='rbf',degree=3,gamma='auto',coef0=0.0,
        shrinking=True,probability=False,tol=0.001,cache_size=200,
        class_weight=None,verbose=False,max_iter=-1,
        decision_function_shape=None,random_state=None)
svc.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=svc.predict(X_test.astype('int'))
a=svc.score(X_train.astype('int'),Y_train.astype('int'))
print("SVC:",a)

# #2.3 NuSVC-支持向量回归
# nusvc=NuSVC(nu=0.5,kernel='rbf',degree=3,
#             gamma='auto',coef0=0.0,shrinking=True,tol=0.001,
#             cache_size=200,verbose=False,max_iter=-1,probability=False,
#             decision_function_shape=None,random_state=None)
# nusvc.fit(X_train.astype('int'),Y_train.astype('int'))
# Y_pred=nusvc.predict(X_test.astype('int'))
# a=nusvc.score(X_train.astype('int'),Y_train.astype('int'))
# print("NuSVC:",a)

#2.4 LinearSVR-线性支持向量回归
linearsvr=LinearSVR(epsilon=0.0,tol=0.0001,C=1.0,
                    loss='epsilon_insensitive',
                    fit_intercept=True,intercept_scaling=1.0,
                    dual=True,verbose=0,random_state=None,
                    max_iter=1000)
linearsvr.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=linearsvr.predict(X_test.astype('int'))
a=linearsvr.score(X_train.astype('int'),Y_train.astype('int'))
print("LinearSVR:",a)

#2.5 SVR-支持向量回归
svr=SVR(kernel='rbf',degree=3,gamma='auto',coef0=0.0,tol=0.001,
        C=1.0,epsilon=0.1,shrinking=True,cache_size=200,verbose=False,
        max_iter=-1)
svr.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=svr.predict(X_test.astype('int'))
a=svr.score(X_train.astype('int'),Y_train.astype('int'))
print("SVR:",a)

#2.6 NuSVR-支持向量回归
nusvr=NuSVR(nu=0.5,C=1.0,kernel='rbf',degree=3,
            gamma='auto',coef0=0.0,shrinking=True,tol=0.001,
            cache_size=200,verbose=False,max_iter=-1)
nusvr.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=nusvr.predict(X_test.astype('int'))
a=nusvr.score(X_train.astype('int'),Y_train.astype('int'))
print("NuSVR:",a)

#2.7 OneClassSVM
oneclasssvm=OneClassSVM(kernel='rbf',degree=3,gamma='auto',
                        coef0=0.0,tol=0.001,nu=0.5,shrinking=True,
                        cache_size=200,verbose=False,max_iter=-1,random_state=None)
oneclasssvm.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=oneclasssvm.predict(X_test.astype('int'))
# a=oneclasssvm.score(X_train.astype('int'),Y_train.astype('int'))
# print("OneClassSVM:",a)
'''

'''
####################################################################################Gaussian Naive Bayes
##三、贝叶斯模型
#3.1 GaussianNB-高斯贝叶斯分类器
gaussian=GaussianNB()
gaussian.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=gaussian.predict(X_test.astype('int'))
a=gaussian.score(X_train.astype('int'),Y_train.astype('int'))
b=gaussian.predict_proba(X_train)
c=gaussian.predict_log_proba(X_train)
print("Gaussian Naive Bayes:",a,b,c)

# #3.2 MultinomialNB-多项式贝叶斯分类器
# multinomialnb=MultinomialNB(alpha=1.0,fit_prior=True,class_prior=None)
# multinomialnb.fit(X_train.astype('int'),Y_train.astype('int'))
# Y_pred=multinomialnb.predict(X_test.astype('int'))
# a=multinomialnb.score(X_train.astype('int'),Y_train.astype('int'))
# b=multinomialnb.predict_proba(X_train)
# c=multinomialnb.predict_log_proba(X_train)
# print("MultinomialNB:",a,b,c)

#3.3 BernoulliNB-伯努利贝叶斯分类器
bernoullinb=BernoulliNB(alpha=1.0,binarize=0.0,fit_prior=True,
                        class_prior=None)
bernoullinb.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=bernoullinb.predict(X_test.astype('int'))
a=bernoullinb.score(X_train.astype('int'),Y_train.astype('int'))
b=bernoullinb.predict_proba(X_train)
c=bernoullinb.predict_log_proba(X_train)
print("BernoulliNB:",a,b,c)
'''

'''
##################################################################################Random Forest
#四、决策树
#4.1 DecisionTreeRegressor-回归决策树
# decisiontree=DecisionTreeClassifier(criterion='mse',splitter='best',
#                                     max_depth=None,min_impurity_split=2,
#                                     min_samples_leaf=1,min_weight_fraction_leaf=0.0,
#                                     max_features=None,random_state=None,
#                                     max_leaf_nodes=None,presort=False)
# decisiontree.fit(X_train.astype('int'),Y_train.astype('int'))
# Y_pred=decisiontree.predict(X_test.astype('int'))
# a=decisiontree.score(X_train.astype('int'),Y_train.astype('int'))
# print("Decision Tree Regressor:",a)

#4.2 DecisionTreeClassifier-分类决策树
decisiontree_classifier=DecisionTreeClassifier(criterion='gini',
                                               splitter='best',
                                               max_depth=None,
                                               min_impurity_split=2,
                                               min_samples_leaf=1,
                                               min_weight_fraction_leaf=0.0,
                                               max_features=None,random_state=None,
                                               max_leaf_nodes=None,class_weight=None,presort=False)
decisiontree_classifier.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=decisiontree_classifier.predict(X_test.astype('int'))
a=decisiontree_classifier.score(X_train.astype('int'),Y_train.astype('int'))
b=decisiontree_classifier.predict_proba(X_train)
c=decisiontree_classifier.predict_log_proba(X_train)
print("BernoulliNB:",a,b,c)
'''

'''
####################################################################################KNN
#五、KNN
#5.1 KNeighborsClassifier-knn分类模型
knn=KNeighborsClassifier(n_neighbors=5,weights='uniform',
                         algorithm='auto',leaf_size=30,
                         p=2,metric='minkowski',metric_params=None,
                         n_jobs=1)
knn.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=knn.predict(X_test.astype('int'))
a=knn.score(X_train.astype('int'),Y_train.astype('int'))
b=knn.predict_proba(X_train)
c=knn.kneighbors()
d=knn.kneighbors_graph()
print("KNN:",a,b)

#5.2 KNeighborsRegressor-knn回归模型
knn_regressor=KNeighborsRegressor(n_neighbors=5,weights='uniform',
                                  algorithm='auto',leaf_size=30,p=2,
                                  metric_params=None,metric='minkowski',n_jobs=1)
knn_regressor.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=knn_regressor.predict(X_test.astype('int'))
a=knn_regressor.score(X_train.astype('int'),Y_train.astype('int'))
c=knn_regressor.kneighbors()
d=knn_regressor.kneighbors_graph()
print("KNN Regressor:",a,c,d)
'''

'''
##################################################################################################AdaBoost
#六、AdaBoost
#6.1 AdaBoostClassifier-分类器
AdaBoostclassifier=AdaBoostClassifier(base_estimator=None,n_estimators=50,
                            learning_rate=1.0,algorithm='SAMME.R',
                            random_state=None)
AdaBoostclassifier.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=AdaBoostclassifier.predict(X_test.astype('int'))
a=AdaBoostclassifier.score(X_train.astype('int'),Y_train.astype('int'))
b=AdaBoostclassifier.predict_proba(X_train)
c=AdaBoostclassifier.predict_log_proba(X_train)
d=AdaBoostclassifier.staged_predict(X_train)
e=AdaBoostclassifier.staged_predict_proba(X_train)
f=AdaBoostclassifier.staged_score(X_train,Y_train)
print("Adaboost Classifier:",a,b,c,d,e,f)

#6.2 AdaBoostRegressor-回归器
AdaBoostregressor=AdaBoostRegressor(base_estimator=None,n_estimators=50,
                                    learning_rate=1.0,loss='linear',random_state=None)
AdaBoostregressor.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=AdaBoostregressor.predict(X_test.astype('int'))
a=AdaBoostregressor.score(X_train.astype('int'),Y_train.astype('int'))
b=AdaBoostregressor.staged_predict(X_train)
c=AdaBoostregressor.staged_score(X_train,Y_train)
print("Adaboost Classifier:",a,b,c)
'''

'''
##############################################################
#七、梯度提升树
#7.1 GradientBoostingClassifier-GBDT分类模型
gbdt=GradientBoostingClassifier(loss='deviance',learning_rate=0.1,
                                n_estimators=100,subsample=1.0,
                                min_samples_split=2,min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,max_depth=3,init=None,
                                random_state=None,max_features=None,verbose=0,max_leaf_nodes=None,
                                warm_start=False,presort='auto')
gbdt.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=gbdt.predict(X_test.astype('int'))
a=gbdt.score(X_train.astype('int'),Y_train.astype('int'))
b=gbdt.predict_proba(X_train)
c=gbdt.predict_log_proba(X_train)
d=gbdt.staged_predict(X_train)
e=gbdt.staged_predict_proba(X_train)
print("Adaboost Classifier:",a,b,c,d,e)

#7.2 GradientBoostingRegressor-GBDT回归模型
gbdtregressor=GradientBoostingRegressor(loss='ls',learning_rate=0.1,
                                n_estimators=100,subsample=1.0,
                                min_samples_split=2,min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,max_depth=3,init=None,
                                random_state=None,max_features=None,verbose=0,max_leaf_nodes=None,
                                warm_start=False,presort='auto')
gbdtregressor.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=gbdtregressor.predict(X_test.astype('int'))
a=gbdtregressor.score(X_train.astype('int'),Y_train.astype('int'))
b=gbdtregressor.staged_predict(X_train)
print("Adaboost Classifier:",a,b)
'''

'''
########################################################################
#八、Random Forest
#8.1 RandomForestClassifier-随机森林分类模型
random_forest=RandomForestClassifier(n_estimators=10,criterion='gini',
                                     max_depth=None,min_samples_split=2,
                                     min_samples_leaf=1,min_weight_fraction_leaf=0.0,
                                     max_features='auto',max_leaf_nodes=None,bootstrap=True,
                                     oob_score=False,n_jobs=1,random_state=None,verbose=0,warm_start=False,
                                     class_weight=None)
random_forest.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=random_forest.predict(X_test.astype('int'))
a=random_forest.score(X_train.astype('int'),Y_train.astype('int'))
b=random_forest.predict_log_proba(X_train)
c=random_forest.predict_proba(X_train)
print("Random Forest Classifier:",a,b,c)

#8.2 RandomForestRegressor-随机森林回归模型
random_forest_Regressor=RandomForestRegressor(n_estimators=10,criterion='mse',
                                     max_depth=None,min_samples_split=2,
                                     min_samples_leaf=1,min_weight_fraction_leaf=0.0,
                                     max_features='auto',max_leaf_nodes=None,bootstrap=True,
                                     oob_score=False,n_jobs=1,random_state=None,verbose=0,warm_start=False)
random_forest_Regressor.fit(X_train.astype('int'),Y_train.astype('int'))
Y_pred=random_forest_Regressor.predict(X_test.astype('int'))
a=random_forest_Regressor.score(X_train.astype('int'),Y_train.astype('int'))
print("Random Forest Regressor:",a)
'''
