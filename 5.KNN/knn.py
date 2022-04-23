#knn
'''
    TODO:2022.4.9 2020211948蒋佳男 learning progess
'''

from numpy import *
import operator
def createDataSet():
    group = array([[1.0,1.1],
                   [1.0,1.0],
                   [0,0],
                   [0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(vecX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(vecX,(dataSetSize, 1)) - dataSet  #每个点对应坐标之差
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)     #这一步求平方和
    distances = sqDistances**0.5            #这一步就是开根号
    sortedDistances = distances.argsort()   #这里面保存的是下标值，对应的下表就是对应的label值
    classCount = {}                         #结果字典，下面对这个字典进行排序
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        if voteIlabel not in classCount.keys():
            classCount[voteIlabel] = 0
        else:
            classCount[voteIlabel] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda x : x[1], reverse=True) #注意返回值是list，然后每个元素是元组
    return sortedClassCount[0][0]           #返回标签值即可

