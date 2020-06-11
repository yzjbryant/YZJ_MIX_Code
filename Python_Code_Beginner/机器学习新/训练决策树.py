#In[1]:
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

X_train,X_test,y_train,y_test=train_test_split(X,y)
pipeline=Pipeline([('clf',DecisionTreeClassifier(criterion='entropy'))])
parameters={
    'clf_max_depth':(150,155,160),
    'clf_min_samples_split':(2,3),
    'clf_max_samples_leaf':(1,2,3)
}
grid_search=GridSearchCV(pipeline,parameters,n_j,verbose=1,scoring='f1')
grid_search.fit(X_train,y_train)
best_parameters=grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('t%s:%er'%(param_name,best_parameters[param_name]))

predictions=grid_search.predict(X_test)
print(classification_report(y_test,predictions))

obs=-1