from sklearn import preprocessing
enc=preprocessing.OneHotEncoder()
enc.fit([[0,0,3],[1,1,0],[0,2,1],[1,0,2]])
array=enc.transform([[0,1,3]]).toarray()
print(array)