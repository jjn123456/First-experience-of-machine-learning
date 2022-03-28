from numpy import *
from math import *

'''
def randCent(dataMat, k):
    """利用矩阵matrix，优化randCent"""
    n = shape(dataMat)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataMat[:,j])
        rangeJ = float(max(dataMat[:,j]) - minJ)
        centroids[:,j] = minJ + random.rand(k,1) * rangeJ
    return centroids
'''

def loadDataSet(filename):
    """读文件，并且生成所有点坐标的二维集合"""
    dataMat = []
    fltLine = [0.0,0.0]
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine[0] = float(curLine[0])
        fltLine[1] = float(curLine[1])
        dataMat.append(fltLine[:])
    return dataMat

def distEclud(vecA, vecB):
    """计算Euclid距离"""
    n = len(vecA)
    ans = 0.0
    for i in range(n):
        ans += pow(vecA[i] - vecB[i], 2)
    return sqrt(ans)

def findColMin(dataMat, col:int):
    row = len(dataMat)
    columns = []
    for i in range(row):
        columns.append(dataMat[i][col])
    return min(columns)

def findColMax(dataMat, col:int):
    row = len(dataMat)
    columns = []
    for i in range(row):
        columns.append(dataMat[i][col])
    return max(columns)

def randCent(dataMat, k):
    """初始设定 K 个点(超参数)"""
    n = len(dataMat[0])
    centroids = [[0]*n for _ in range(k)]
    for j in range(n):
        minJ = findColMin(dataMat, j)
        rangeJ = float(findColMax(dataMat, j) - minJ)
        for i in range(k):
            centroids[i][j] = float(minJ + rangeJ * random.rand())
    return centroids

def k_means(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = len(dataSet)
    clusterAssment = [0] * m #储存每个点的簇分配结果
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j],dataSet[i])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i] != minIndex:    #说明需要继续循环
                clusterChanged = True
                clusterAssment[i] = minIndex
        print(centroids) #打印了迭代的过程
        #更新质点的位置
        sumClusters = [[0.0,0.0,0.0] for i in range(k)]
        #clusterAssment[i] 对应的是k
        for i in range(m):
            sumClusters[clusterAssment[i]][0] += dataSet[i][0]
            sumClusters[clusterAssment[i]][1] += dataSet[i][1]
            sumClusters[clusterAssment[i]][2] += 1.0
        for i in range(k):
            centroids[i][0] = sumClusters[i][0] / sumClusters[i][2]
            centroids[i][1] = sumClusters[i][1] / sumClusters[i][2]
    return centroids, clusterAssment




dataSet = loadDataSet('testSet.txt')
myCentroids, clustAssing = k_means(dataSet, 4)

#上面的代码正确的到了 K 个组的类质心，根据 clustAssing 数组记录的分组情况可以找到对应的成员点，暂时找省略成员点这一步
print('----------------')
print(myCentroids)
print(clustAssing)