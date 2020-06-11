from numpy import *
import operator

def create_DataSet():
    group = array([1.0,1.1],[1.0,1.0],[0,0],[0,0.1]) #数据集
    labels = ['A','A','B','B'] #标签
    return group, labels

group, labels = kNN.createDataSet()

print(group)
print(labels)

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    #① （以下三行）距离计算
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDisIndicies = distances.argsort()
    classCount = {}
    #②（以下两行）选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),#3.排序
                                  key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#预测分类
kNN.classify0([0,0],group,labels,3)

