#In[1]:
import pandas as pd
df=pd.read_csv('./SMSSSpamCollection',delimiter='t',header=None)
print(df.head())

print('Number of spam messages:%s'%df[df[0]=='spam'][0].count())
print('Number of ham messages:%s'%df[df[0]=='ham'][0].count())

#In[2]:
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score

X=df[1].values
y=df[0].values
X_train_raw,X_test_raw,y_train,y_test=train_test_split(X,y)
vectorizer=TfidfVectorizer()
X_train=vectorizer.fit_transform(X_train_raw)
X_test=vectorizer.transform(X_test_raw)
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
predictions-classifier.predict(X_test)
for i,prediction in enumerate(predictions[:5]):
    print('Predicted:%s,message:%s'%(prediction,X_test_raw[i]))

#In[3]:二元分类性能指标
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

y_test=[0,0,0,0,0,1,1,1,1,1]
y_pred=[0,1,0,0,0,0,0,1,1,1]
confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('Confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.colorbar()
plt.show()

#In[4]:准确率
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

df=pd.read_csv('./sms.csv')
X_train_raw,X_test_raw,y_train,y_test=train_test_split(df['message'],
                                                       df['label'],random_state=11)
vectorizer=TfidfVectorizer()
X_train=vectorizer.fit_transform(X_train_raw)
X_test=vectorizer.transform(X_test_raw)
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
scores=cross_val_score(classifier,X_train,y_train,cv=5)
print('Accuracies:%s'%scores)
print('Mean accuracy:%s'%np.mean(scores))


#In[5]: 精准率和召回率和F1
precisions=cross_val_score(classifier,X_train,y_train,cv=5,scoring='precision')
print('Precision:%s'%np.mean(precisions))

recalls=cross_val_score(classifier,X_train,y_train,cv=5,scoring='recall')
print('Recall:%s'%np.mean(recalls))

fls=cross_val_score(classifier,X_train,y_train,cv=5,scoring='f1')
print('F1 score:%s'%np.mean(fls))

predictions=classifier.predict_proba(X_test)
false_positive_rate,recall,thresholds=roc_curve(y_test,predictions[:,1])
roc_auc=auc(false_positive_rate,recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,recall,'b',label='AUC=%.2f'%roc_auc)
plt.legend(loc='low right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()

#In[1]:使用网格搜索微调模型
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
# from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#In[1]:多类别分类


