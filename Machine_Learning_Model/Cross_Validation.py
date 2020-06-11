from sklearn.model_selection import KFold,cross_val_score

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100)

cv=KFold(len(train),n_folds=10,indices=False)
results=[]
for traincv,testcv in cv:
    probas=model.fit(train[traincv],target[traincv]).predict_proba(train[testcv])
    results.append(Error_function)
print("Result: "+str(np.array(results).mean()))
