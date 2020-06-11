import numpy as np
from sklearn.model_selection import cross_val_score,ShuffleSplit
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# boston=load_boston()
# X=boston["data"]
# Y=boston['target']
# names=boston["feature_names"]

data=pd.read_excel('CIF_Parameters.xlsx',header=0)
X=data.drop(['Materials','Loading_C4H4S_298K_10kpa'],axis=1)
Y=data['Loading_C4H4S_298K_10kpa']
label=list(X.columns.values)

rf=RandomForestRegressor(n_estimators=20,n_jobs=-1,max_depth=4)
scores=[]
for i in range(X.shape[1]):
    score=cross_val_score(rf,X.values[:,i:i+1],Y,scoring="r2",cv=ShuffleSplit(len(X),3,.3))
    scores.append((round(np.mean(score),3),label[i]))
print(sorted(scores,reverse=True))