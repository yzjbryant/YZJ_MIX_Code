#函数img2vector 将图像转化为向量
def img2vextor(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i + j] = int(lineStr[j])
    return returnVect

testVector = kNN.img2vector('testDigits/0_13.txt')
print(testVector[0,0:31])
print(testVector[0,32:63])

#手写数字识别系统的测试代码
def handwritingClassTest():
    hwlabels = []
    trainingFileList = listdir('trainingDigits') #1.获取目录内容
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        #2.（以下三行）从文件名解析分类数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('-')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vextor('trainingDigits/%s'%fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('-')[0])
        vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with:%d,the real answer is :%d"%(classifierResult,vectorUnderTest))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is : %d" %errorCount)
    print("\nthe total error rate is: %f "% (errorCount/float(mTest)))

print(kNN.handwritingClassTest())
