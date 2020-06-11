#在Python中使用Matplotlib注解绘制树形图
#annotations注解工具，在图形上添加文本注释
#使用文本注解绘制树节点
import matplotlib.pyplot as plt
#1.(以下三行)定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle='<-')

#2.(以下两行)绘制带箭头的注解
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt,textcoords='axes fraction'
                            va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode('决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
#plotNode函数执行绘图功能
#createPlot函数创建一个新图形并清空绘图区

import treePlotter
print(treePlotter.createPlot())

#getNumLeafs()、getTreeDepth()来获取叶节点的数目和树的层数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        #1.（以下三行）测试节点的数据类型是否为字典
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.key()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

#type()函数可以判断子节点是否为字典类型

def retrieveTree(i):
    listOfTrees = [{'no surfacing':{0:'no',1:{'flippers':\
                                                  {0:'no',1:'yes'}}}},
                   'no surfacing':{0:'no',1:{'flippers':\
                                                     {0:{'head':{0:'no',1:'yes'}},1:'no'}}}]
    return listOfTrees

reload(treePlotter)
print(treePlotter.retrieveTree(1))
print(myTree = treePlotter.retrieveTree(0))
print(treePlotter.getNumLeafs(myTree))
print(treePlotter.getTreeDepth(myTree))

#retrieveTree()函数主要用于测试，返回预定义的树结构

#plotTree函数
#1.（以下四行）在父子节点间填充文本信息
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString,va="center",ha="center",rotat)

def plotTree(myTree,parentPt,nodeTxt):
    #2.(以下两行)计算宽和高
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalw,plot)
    #3.标记子节点属性值
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    #4.（以下两行）减小y偏移
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))  #recursion
    else:
        plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
        plotNode(secondDict[key],(plotTree.xOff,plotTrr.yOff),,cntrPt,leaf...)
        plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))

plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1,facecolor = 'white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW;plotTree.yOff = 1.0;
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

reload(treePlotter)
print(myTree=treePlotter.retrieveTree(0))
print(treePlotter.createPlot(myTree))

print(myTree['no surfacing'][3]='maybe')
print(myTree)
print(treePlotter.createPlot(myTree))

#使用决策树的分类函数(递归函数)
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    #1.将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel
print(myDat,labels=trees.createDataSet())
print(labels)
print(myTree=treePlotter.retrieveTree(0))
print(myTree)
print(trees.classify(myTree,labels,[1,0]))
>>>'no'
print(tree.classify(myTree,labels,[1,1]))
>>>'yes'

#使用pickle模块存储决策树
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

print(trees.storeTree(myTree,'classifierStorage.txt'))
print(trees.grabTree('classifierStorage.txt'))




























