#datingTestSet.txt 每个样本数据占一行，共1000行
def file2matrix(filename):
    fr = open(filename)
    arrayolines = fr.readlines()
    numberOfLines = len(arrayOlines) #1.得到文件行数
    returnMat = zeros((numberOfLines,3)) #2.创建返回的Numpy矩阵
    classLabelVetor = []
    index = 0
    #3.（以下三行）解析文件数据到列表
    for line in arrayOlines:
        line = line.strip() #截取掉所有回车字符
        listFromLine = line.split('\t') #tab字符将上一步得到的整行数据分割成一个元素列表
        returnMat[index,:] = listFromLine[0:3] #选取前3个元素存储到特征矩阵中
        classLabelVetor.append(int(listFromLine[-1])) #索引值-1表示列表中的最后一列元素
        index += 1
    return returnMat,classLabelVetor

reload(kNN)
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')

print(datingDataMat)
print(datingLabels)

#分析数据，使用Matplotlib创建散点图
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
plt.show()

#数值归一化 将任意取值范围的特征值转化为0到1区间内的值
newValue = (oldValue - min)/(max - min)

#函数autoNorm可以自动将数字特征值转化为0到1的区间
#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0) #参数0使得函数可以从列中选取最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1)) #tile函数将变量内容复制成输入矩阵同样大小的矩阵
    normDataSet = normDataSet/tile(ranges,(m,1)) #1.特征值相除
    return normDataSet,ranges,minVals

#矩阵除法需要使用函数linalg.solve(matA,matB)
reload(kNN)
normMat,ranges,minVals = kNN.autoNorm(datingDataMat)
print(normMat)
print(ranges)
print(minVals)

#分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                     datingLabels[numTestVecs:m],3)
        print("the classifier came back with:%d,the real answer is:%d"\
              %(classifierResult,datingLabels[i])
            if (classifierResult != datingLabels[i]):
                errorCount += 1.0
        print("the total error rate is:%f" % (errorCount/float(numTestVecs)))

#classify0分类器函数
print(kNN.datingClassTest())

#约会网站预测函数
#函数raw_input()允许用户输入文本行命令并返回用户所输入内容
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input(\
        "percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0(inArr-\
                                 minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person: ",\
          resultList[classifierResult - 1])
print(kNN.classifyPerson())




