# #In[1]:
# # %matplotlib inline
#
# #In[2]:
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
#
# X,y=make_classification(n_samples=1000,n_features=50,n_informative=30,n_clusters_per_class=3,random_state=11)
# X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=11)
#
# clf=DecisionTreeClassifier(random_state=11)
# clf.fit(X_train,y_train)
# print('Decision tree accuracy:%s'%clf.score(X_test,y_test))
#
# #In[3]:
# #When an argument for the base_estimator parameter is not passed, the default DecisionTreeClassifier is used
# clf=AdaBoostClassifier(n_estimators=50,random_state=11)
# clf.fit(X_train,y_train)
# accuracies.append(clf.score(X_test,y_test))
#
# plt.title('Ensemble Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Number of base estimate in ensemble')
# plt.plot(range(1,51),[accuracy for accuracy in clf.staged_score(X_test,y_test)])

#堆叠法
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.base import clone,BaseEstimator,TransformerMixin,ClassifierMixin

class StackingClassifier(BaseEstimator,TransformerMixin,ClassifierMixin):

    def __init__(self,classifiers):
        self.classifiers=classifiers
        self.meta_classifier=DecisionTreeClassifier()

    def fit(self,X,y):
        for clf in self.classifiers:
            clf.fit(X,y)
        self.meta_classifier.fit(self._get_meta_features(X),y)
        return self

    def _get_meta_features(self,X):
        probas=np.asarray([clf.predict_proda(X) for clf in self.classifiers])
        return np.concatenate(probas,axis=1)

    def predict(self,X):
        return self.meta_classifier.predict(self._get_meta_features(X))

    def predict_proda(self,X):
        return self.meta_classifier.predict_proba(self._get_meta_features(X))

X,y=make_classification(n_samples=1000,n_features=50,n_informative=30,n_clusters_per_class=3,random_state=11)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=11)

lr=LogisticRegression()
lr.fit(X_train,y_train)
print('Logistic regression accurucy:%s'%lr.score(X_test,y_test))

knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train,y_train)
print('KNN accuracy:%s'%knn_clf.score(X_test,y_test))

base_classifiers=[lr,knn_clf]
stacking_clf=StackingClassifier(base_classifiers)
stacking_clf.fit(X_train,y_train)
print('Stacking classifier accuracy:%s'%stacking_clf.score(X_test,y_test))


