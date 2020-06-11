from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

data=np.random.randn(15,5)

pca=PCA()

pca.fit(data)

#模型的各个特征向量
a=pca.components_
print(a)

#各个成分各自的方差百分比（贡献率）
b=pca.explained_variance_ratio_
print(b)

#重新建立PCA模型
pca=PCA(3)
pca.fit(data)
low_d=pca.transform(data) #降维
pd.DataFrame(low_d).to_excel('result.xlsx')
pca.inverse_transform(low_d)  #恢复数据

