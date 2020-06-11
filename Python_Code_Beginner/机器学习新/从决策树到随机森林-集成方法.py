# #In[1]:
# import numpy as np
# #Sample 10 integers
# sample=np.random.randint(low=1,high=100,size=10)
# # print(sample)
# print("Original sample:%s"%sample)
# print("Sample mean:%s"%sample.mean())
#
# #Bootstrap re-sample 100 times by re-sampling with replacing from the original sample
# #引导重新采样100次，通过重新采样替换原来的样本
# resamples=[np.random.choice(sample,size=sample.shape) for i in range(100)]
# print('Number of bootstrop re-samples:%s'%len(resamples))
# print('Example re-sample:%s'%resamples[0])
#
# resample_means=np.array([resample.mean() for resample in resamples])
# print('Mean of re-samples\' means: %s'% resample_means.mean())



#scikit-learn训练随机森林
#In[1]:
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X,y=make_classification(
    n_samples=1000,n_features=100,n_informative=20,n_clusters_per_class=2,random_state=11
)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=11)

clf=DecisionTreeClassifier(random_state=11)
clf.fit(X_train,y_train)
predictions=clf.predict(X_test)
print(classification_report(y_test,predictions))

#In[2]:
clf=RandomForestClassifier(n_estimators=10,random_state=11)
clf.fit(X_train,y_train)
predictions=clf.predict(X_test)
print(classification_report(y_test,predictions))


