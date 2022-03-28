import math
import random
import matplotlib.pyplot as plt
import numpy as np
import random
from numpy import mat

class Point:
    def __init__(self,x,y,statu):
        self.x = x
        self.y = y
        self.statu = statu
        self.visited = False
        self.cluster = -1

    def get_statu(self) ->str :
        return self.statu

    def get_x_and_y(self) -> int:
        return self.x, self.y

    def to_string(self):
        return '({}, {})_{}_cluster: {}'.format(self.x,self.y,self.statu,self.cluster)

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

def getScatterfromSet(dataMat):
    '''将数据的x，y坐标分别提取出来 -> 分别放在两个数组或矩阵中'''
    x = []
    y = []
    n = len(dataMat)
    for temp in dataMat:
        x.append(temp[0])
        y.append(temp[1])
    return x,y

def get_distance(Point1,Point2) -> float :
    x1 = Point1.x
    y1 = Point1.y
    x2 = Point2.x
    y2 = Point2.y
    return math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))

def count_visited(core_points) -> int :
    ans = 0
    for i in core_points:
        if i.visited == True:
            ans = ans + 1
    return ans

def points_init():
    mydata = loadDataSet('testSet.txt')
    n = len(mydata)
    dataPoints = [Point(-1, -1, '') for i in range(n)]
    for i in range(n):
        dataPoints[i] = Point(mydata[i][0], mydata[i][1], 'noise')
    return dataPoints,n

'''下面开始设置超参数 ε(raduis) and MinPts'''
#ε
Eps = 1.5
MinPts = 11.0
dataPoints,length = points_init()#初始化
for i in range(length):#标记核心点
    count = 0
    for j in range(length):
        if j == i:
            continue
        if get_distance(dataPoints[i],dataPoints[j]) < Eps:
            count = count + 1
            if count >= MinPts:
                dataPoints[i].statu = 'corept'
                break

for i in range(length):#标记非核心点, 剩下的都是noise
    if dataPoints[i].statu == 'corept':
        continue
    for j in range(length):
        if i == j:
            continue
        if get_distance(dataPoints[i],dataPoints[j]) < Eps and dataPoints[j].statu == 'corept':
            dataPoints[i].statu = 'noncorept'
            break;

#1.处理核心点
core_points = []
for t in dataPoints:           #核心点数组
    if t.statu == 'corept':
        core_points.append(t)

length2 = len(core_points)
index = 1

for i in range(length2):        #1.处理核心点
    if core_points[i].visited == False:
        core_points[i].visited = True
        core_points[i].cluster = index
        index = index + 1
        a = []
        a.append(core_points[i])
        while len(a) != 0:
            temp = a.pop(0)
            for j in range(length2):
                if j == i or core_points[j].visited == True:
                    continue
                if get_distance(core_points[i],core_points[j]) < Eps :
                    a.append(core_points[j])
                    core_points[j].visited = True
                    core_points[j].cluster = temp.cluster

colors = ['b','r','m','k','g','c','y','w']
#2.处理非核心点, noise点不用管
non_core_points = []
for i in dataPoints:
    if i.statu == 'noncorept':
        non_core_points.append(i)

length2 = len(non_core_points)
for i in range(length2):
    for temp in core_points:
        if get_distance(non_core_points[i],temp) < Eps:
            non_core_points[i].cluster = temp.cluster
            break

#下面开始cluster分类
for i in range(length):
    if dataPoints[i].cluster == -1:
        plt.scatter(dataPoints[i].x,dataPoints[i].y,color='grey')
    elif dataPoints[i].cluster != -1:
        plt.scatter(dataPoints[i].x, dataPoints[i].y,color=colors[dataPoints[i].cluster-1])
plt.show()




