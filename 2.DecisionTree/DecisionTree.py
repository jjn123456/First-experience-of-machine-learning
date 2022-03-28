from math import log
import operator

def calShanonEnt(dataSet):
    '''计算整个数据集的原始香农熵'''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0   #加入集合
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def spiltDataSet(dataSet, axis, value):
    '''按照给定的特征划分数据集'''
    retDataSet = []
    for featVec in dataSet:                         #featVec是一个List
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])  #拆List的包，然后把剩余元素加进去
            retDataSet.append(reduceFeatVec)        #加List
    return retDataSet

def chooceBestFeatureToSplit(dataSet):
    '''选择最好的数据集划分方式'''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calShanonEnt(dataSet)
    bestInfoGain = 0.0  #最好的信息增益
    bestFeature = -1    #最好属性对应的下标
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]       #某属性对应的一列值被纳入featList
        uniqueVals = set(featList)                           #利用set来去重
        newEntropy = 0.0
        for value in uniqueVals:
            subtractDataset = spiltDataSet(dataSet, i, value)
            prob = len(subtractDataset)/float(numFeatures + 1) #除以行数
            newEntropy += prob * calShanonEnt(subtractDataset)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            #更行bestInfoGain
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList) -> str:
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #classCount.items()这是个List,这句话相当于对二维数组按照第二维进行排序
    return sortedClassCount[0][0] #二维数组的第一维是string

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:#没有经过第一步
        return majorityCnt(classList)
    bestFeat = chooceBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(spiltDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
labels = ['no surfing','flippers']
tree = createTree(dataSet,labels)
print(tree)
