# #In [1]:
# from sklearn.feature_extraction import DictVectorizer
# one_hot_encoder=DictVectorizer()
# X=[
#     {'city':'New Work'},
#     {'city':'San Francisco'},
#     {'city':'Chapel Hill'}
# ]
# print(one_hot_encoder.fit_transform(X).toarray())

#In[1]: 特征标准化
# from sklearn import preprocessing
# import numpy as np
# X=np.array([
#     [0.,0.,5.,13.,9.,1.],
#     [0.,0.,13.,15.,10.,15.],
#     [0.,3.,15.,2.,0.,11.]
# ])
# print(preprocessing.scale(X))

#In[1]:词袋模型
# corpus=[
#     'UNC played Duke in basketball',
#     'Duke lost the basketball game'
# ]
# #In[2]:
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer=CountVectorizer()
# print(vectorizer.fit_transform(corpus).todense())
# print(vectorizer.vocabulary_)
#
# #In[3]:
# corpus.append('I ate a sandwich')
# print(vectorizer.fit_transform(corpus).todense())
# print(vectorizer.vocabulary_)
#
# #In[4]:
# from sklearn.metrics.pairwise import euclidean_distances
# X=vectorizer.fit_transform(corpus).todense()
# print('Distance between 1st and 2nd documents:',
#       euclidean_distances(X[0],X[1]))
# print('Distance between 1st and 3rd documents:',
#       euclidean_distances(X[0],X[2]))
# print('Distance between 2nd and 3rd documents:',
#       euclidean_distances(X[1],X[2]))
#
# #In[5]:
# vectorizer=CountVectorizer(stop_words='english')
# print(vectorizer.fit_transform(corpus).todense())
# print(vectorizer.vocabulary_)

#In[6]:
# corpus=[
#     'He ate the sandwiches',
#     'Every sandwich was eaten by him'
# ]
# vectorizer=CountVectorizer(binary=True, stop_words='english')
# print(vectorizer.fit_transform(corpus).todense())
# print(vectorizer.vocabulary_)

#In[7]: 词形还原
# corpus=[
#     'I am gathering ingredients for the sandwich',
#     'There were many wizards at the gathering'
# ]
#
# #In[8]:
# from nltk.stem.wordnet import WordNetLemmatizer
# lemmatizer=WordNetLemmatizer()
# print(lemmatizer.lemmatize('gathering','v'))
# print(lemmatizer.lemmatize('gathering','n'))
#
# #In[9]:
# from nltk.stem import PorterStemmer
# stemmer=PorterStemmer()
# print(stemmer.stem('gathering'))

#In[1]: 词形还原
# from nltk import word_tokenize
# from nltk.stem import PorterStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk import pos_tag
#
# wordnet_tags=['n','v']
# corpus=[
#     'He ate the sandwiches',
#     'Every sandwich was eaten by him'
# ]
# stemmer=PorterStemmer()
# print('Stemmed:', [[stemmer.stem(token) for token in
#                     word_tokenize(document)] for document in corpus])
#
# def lemmatize(token,tag):
#     if tag[0].lower() in ['n','v']:
#         return lemmatizer.lemmatize(token,tag[0].lower())
#     return token
#
# lemmatizer=WordNetLemmatizer()
# tagged_corpus=[pos_tag(word_tokenize(document)) for  document in corpus]
# print('Lemmatized:',[[lemmatize(token,tag) for token, tag in document] for document in tagged_corpus])

#In [1]: 单词频数
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
#
# corpus=['The dog ate a sandwich, the wizard tranfigured a sandwich, and I ate a sandwich']
# vectorizer=CountVectorizer(stop_words='english')
# frequencies=np.array(vectorizer.fit_transform(corpus).todense())[0]
# print(frequencies)
# for token, index in vectorizer.vocabulary_.items():
#     print('The token "%s" appears %s times' % (token,
#                                                frequencies[index]))

#In[1]:
from sklearn.feature_extraction.text import TfidfVectorizer

