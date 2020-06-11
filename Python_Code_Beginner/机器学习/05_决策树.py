import operator
#创建分支的伪代码函数createBranch()
#检测数据集中的每个子项是否属于同一分类：
If so return 类标签：
Else
    寻找划分数据集的最好特征
    划分数据集
    创建分支节点
        for 每个划分的子集
            调用函数createBranch()并增加返回结果到分支节点中
        return 分支节点

#createBranch()是一个递归函数，在倒数第二行直接调用自己

#计算给定数据集的香农熵
from math import log

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    #1.（以下五行）为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        #2.以2为底求对数
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

 def createDataSet():
     dataSet = [[1,1,'yes'],
                [1,1,'yes'],
                [1,0,'no'],
                [0,1,'no'],
                [0,1,'no']]
     labels = ['no surfacing','flippers']
     return dataSet,labels

print(reload(05_决策树.py))
print(myDat,labels=05_决策树.createDataSet())
print(myDat)
#增加第三个名为maybe的分类
print(myDat[0][-1]='maybe')
print(myDat)
print(05_决策树.calcShannonEnt(myDat))

#按照给定特征划分数据集(数据集、特征、需要返回的特征值)
def splitDataSet(dataSet,axis,value):
    #1.创建新的list对象
    retDataSet = []
    for featVec in dataSet:
        #2.（以下三行）抽取
        reduceFeatVec = featVec[:axis]
        reduceFeatVec.extend(featVec[axis+1:])
        retDataSet.append(reduceFeatVec)
    return retDataSet

a=[1,2,3]
b=[4,5,6]
a.append(b)
print(a)
>>>[1,2,3,[4,5,6]]

a.extend(b)
>>>[1,2,3,4,5,6]

reload(05_决策树)
print(myDat,labels=05_决策树.createDataSet())
print(myDat)

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    bestEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        #1.(以下两行)创建唯一的分类标签列表
        featlist = [example[i] for example in dataSet]
        uniqueVals = set(featlist)
        newEntropy = 0.0
        #2.(以下五行)计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataset = splitDataSet(dataSet,i,value)
            prob = len(subDataset)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataset)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            #3.计算最好的信息增益
    bestInfoGain = infoGain
    bestFeature = i
return bestFeature

reload(05_决策树)
print(myDat,labels=05_决策树.createDataSet())
print(myDat)

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(),\
                                  key = operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]

    #创建树的函数代码
    def createTree(dataSet,labels):
        classList = [example[-1] for example in dataSet]
        #1.(以下两行)类别完全相同则停止继续划分
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        #2.(以下两行)遍历完所有特征时返回出现次数最多的
        if len(dataSet[0]) == 1:
            return majorityCnt(classList)
        bestFeat = chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}
        #3.得到列表包含的所有属性值
        del(labels[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = createTree(splitDataSet
                                                      (dataSet,bestFeat,value),subLabels)
        return myTree

reload(05_决策树)
print(myDat,labels=05_决策树.createDataSet())
print(myDat)















