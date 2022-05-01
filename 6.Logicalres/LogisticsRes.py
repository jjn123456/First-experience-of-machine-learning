from numpy import *
import matplotlib.pyplot as plt
def loadDataSet():
    dataList = []
    labelList = []
    fr = open('logicaltestSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataList.append([1.0, float(lineArr[0]), float(lineArr[1])]) #第一维的1.0，是相当于加入了b这个偏置
        labelList.append(int(lineArr[2]))
    return array(dataList), array(labelList).reshape(len(dataList),1)

def sigmoid(x):             #激活函数
    return 1/(1+exp(-x))

def lossfunction(X, y, w):  #如果y是1 yhat也是1(同y=0，yhat=0情况),那么这样损失函数就是最小的
    y_hat = sigmoid(dot(X,w))
    first = y * log(y_hat)
    second = (1-y) * log(1-y_hat)
    return sum(first + second) / len(X) * (-1) #所有样本损失函数的和

def gradientDescent(X,y,w,iters,alpha):
    m = X.shape[0]
    LossArr = []
    for i in range(iters):
        y_hat = sigmoid(dot(X, w))
        w = w - (alpha/m) * dot(X.T, (y_hat-y))
        loss = lossfunction(X,y,w)           #不停迭代就是要让这个损失函数不断变小
        LossArr.append(loss)
        if i % 500 == 0:
            print(loss)
    return w, LossArr

def plotBestFit(w):
    b = w[0,0]
    w1 = w[1,0]
    w2 = w[2,0]
    dataArr, labelArr = loadDataSet()
    n = dataArr.shape[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if labelArr[i] == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s', edgecolors='k')
    ax.scatter(xcord2, ycord2, s=30, c='blue', marker='x')
    x1 = arange(-4., 4., 0.1)
    x2 = (-1) * (w1 * x1 + b)/w2
    ax.plot(x1, x2)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()



a,b = loadDataSet()
w = ones((3,1))
buchang = 0.004
iters = 10000
w_end, lossArr = gradientDescent(a,b,w,iters,buchang)
plotBestFit(w_end)
