from numpy import *
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1] #1代表侮辱性文字，0代表正常言论
    return postingList,classVec


def createVocabList(dataSet):
    #1.创建一个空集
    vocabSet = set([])
    for document in dataSet:
        #2.创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWord2Vec(vocabList,inputSet):
    #3.创建一个其中所含元素都为0的向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

import bayes
listOPosts,listClasses = bayes.loadDataSet()
myVocablist = bayes.createVocabList(listOPosts)
print(myVocablist)

bayes.setOfWord2Vec(myVocablist,listOPosts[0])
bayes.setOfWord2Vec(myVocablist,listOPosts[3])

# 计算每个类别中的文档数目
# 对每篇训练文档：
#     对每个类别：
#         如果词条出现在文档中 -- 增加该词条的计数值
#         增加所有词条的计数值
#     对每个类别：
#         对每个词条：
#             将该词条的数目除以总词条数目得到条件概率
#     返回每个类别的条件概率

#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #1.（以下两行）初始化概率
    p0Num = ones(numWords);
    p1Num = ones(numWords);
    p0Denom = 2.0;
    p1Denom = 2.0;
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            #2.（以下两行）向量相加
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)   #change to log()
    #3.对每个元素做除法
    p0Vect = log(p0Num/p0Denom)   #change to log()
    return p0Vect,p1Vect,pAbusive

from numpy import *
reload(bayes)
listOPosts,listClasses = bayes.loadDataSet()

myVocabList = bayes.createVocabList(listOPosts)
trainMat = []
for postinDoc in listOPosts:
 ...trainMat.append(bayes.setOfWords2Vec(myVocablist,postinDoc))
 ...

>>>p0V,p1V,pAb=bayes.trainNB0(trainMat,listClasses)
>>>pAb
>>>p0V
>>>p1V

#朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    #1.元素相乘
    p1 = sum(vec2Classify * p1Vec)  + log(pClass1)
    p0 = sum(vec2Classify * p0Vec)  + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDOc in listOPosts:
        trainMat.append(setOfwords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

reload(bayes)
print(bayes.testingNB())

#朴素贝叶斯词袋模型
def bagOfWord2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputset:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#构建自己的词列表
mySent = 'This book is the best book on Python or M.L. I have ever laid eyes'
mySent.split()
>>>['This','book','is','the','best','on',...]

#使用正则表示式来切分句子
import re
regEx = re.compile('\\W')
listOfTokens = regEx.split(mySent)
print(listOfTokens)

[tok for tok in listOfTokens if len(tok) > 0]

#.lower()将字符串全部转换为小写
#.upper()将字符串全部转换为大写

emailText = open('email/ham/6.txt').read()
listOfTokens = regEx.split(emailText)

#文件解析及完整的垃圾邮件测试函数
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = [];
    classList = [];
    fullText = [];

    for i in range(1,26):
        #1.（以下七行）导入并解析文本文件
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50);\
    testSet = []

    #2.（以下四行）随机构建训练集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [];\
    trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0

    #3.（以下四行）对测试集分类
    for docIndex in testSet:
        wordVector = sefOfWord2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
        print('the error rate is: ',float(errorCount)/len(testSet))

#测试
print(bayes.spamTest())
print(bayes.spamTest())

import feedparser
ny = feedparser.parse('http://newyork.craiglist.org/stp/index.rss')

ny['entries']
len(ny['entries'])

#RSS源分类器及高频词去除函数
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(),key = operator.itemgetter(1),reveser)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))

    for i in range(minlen):
        #2.每次访问一条RSS源
        wordList = textParse(feed1['entries'][i]['summary'])
        docList = append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed1['entries'][i]['summary'])
        docList = append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    #3.（以下四行）去掉出现次数最高的那些词
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2*minLen)
    testSet=[]

    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(bagOfWord2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0

    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != \
            classList[docIndex]:
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V





































