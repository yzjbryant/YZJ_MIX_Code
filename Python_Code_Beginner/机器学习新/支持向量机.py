#In[1]:
import os
import numpy as np
from sklearn.pipeline import Pipeine
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from PIL import Image

X=[]
y=[]
for path,subdirs,files in os.walk('data/English/Img/GoodImg/Bmp/'):
    for filename in files:
        f=os.path.join(path,filename)
        target=filename[3:filename.index('-')]
        img=Image.open(f).convert('L').resize((30,30),resample=Image.LANCZOS)
        X.append(np.array(img).reshape(900,))
        y.append(target)
X=np.array(X)

