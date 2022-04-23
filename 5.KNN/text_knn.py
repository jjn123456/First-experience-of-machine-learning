
from knn import *
from numpy import *
import numpy as np
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))            #根据数据集的特性可以知道有三行
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()                         #这里主要的目的是去掉 '\n'
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return np.array(returnMat), classLabelVector              #得到处理好的数据集和标签集

def autoNorm(dataSet):                              #归一化特征值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    normMat = np.array(normMat)
    m = normMat.shape[0] #数组长度
    numTestVecs = int(hoRatio * m)
    errorCount = 0
    # 测试向量为normMat[i,:]，数据集为normMat[numTestVecs:m,:]
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('分类器将它分为 {} 类，真实的类为 {}'.format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            #print('mistake is 分类器将它分为 {} 类，真实的类为 {}'.format(classifierResult, datingLabels[i]))
            errorCount += 1.0
    print('最终的错误率为 {:.5f}%'.format(errorCount/float(numTestVecs) * 100))

def classifyPerson():
    resultList = ['一点也不喜欢','可能只有一点点喜欢','很大程度会喜欢']
    percentTats = float(input('有多长时间花费在游戏上: '))
    ffMiles = float(input('每年坐飞机的里程数: '))
    iceCream = float(input('每年会吃多少升的冰淇淋: '))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    clasifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print('你对这个人的喜欢程度: {}'.format(resultList[clasifierResult - 1]))
#text1.
datingClassTest()
#text2.
classifyPerson()
#text3.
#fig = plt.figure()
#ax = fig.add_subplot(111)
#for i in range(datingDataMat.shape[0]):
#    ax.scatter(datingDataMat[i][1], datingDataMat[i][2])
#plt.show()
