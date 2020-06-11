#深度学习用于文本序列

#单词级的one-hot编码简单示例
import numpy as np

samples=['The cat sat on the mat.','The dog ate my homework.']

token_index={}  #构建数据中所有标记的索引
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

max_length=10

results=np.zeros(shape=(len(samples),max_length,max(token_index.values())+1))

for i,sample in enumerate(samples):
    for j,word  in list,enumerate(sample.split())[:max_length]:
        index=token_index.get(word)
        results[i,j,index]=1.


##字符级的one-hot编码
import string

samples=['The cat sat on the mat.','The dog ate my homework.']

characters=string.printable

token_index=dict(zip(range(1,len(characters)+1),characters))

max_length=50

results=np.zeros((len(samples),max_length,max(token_index.keys())+1))

for i,sample in enumerate(samples):
    for j,character in enumerate(sample):
        index=token_index.get(character)
        results[i,j,index]=1.


#用Keras实现单词级的one-hot编码

from keras.preprocessing.text import Tokenizer

samples=['The cat sat on the mat.','The dog ate my homework.']

tokenizer=Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

sequences=tokenizer.texts_to_sequences(samples)

one_hot_results=tokenizer.texts_to_matrix(samples,mode='binary')

word_index=tokenizer.word_index

print('Found %s unique tokens.' %len(word_index))


#将一个Embedding层实例化
from keras.layers import Embedding

embedding_layer=Embedding(1000,64)






















