import  numpy  as  np
import matplotlib.pyplot as plt
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('logicaltestSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #第一维的1.0，是相当于加入了b这个偏置
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def gradAscent(dataMatln, classLabels): #logistic的目标函数相当与要求目标出现的概率(极大似然法)，所以用梯度上升法
    dataMatrix = np.mat(dataMatln)      #批处理所有数据
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001               #目标移动的步长
    maxCycles = 2000             #迭代次数
    weights = np.ones((n,1))    #权重初始为1
    for k in range(maxCycles):  #for循环结束之后得到权重矩阵
        #下面全是矩阵相乘
        y_hat = sigmoid(dataMatrix * weights)   #h是一个列向量   [100*3]*[3*1] = [100*1]
        error = (labelMat - y_hat)              #这里会不会是一个偏移量
        if k % 200 == 0:
            print(sum(error))
        weights = weights + alpha * dataMatrix.transpose() * error   # [3*100]*[100 * 1] = [0.0, 0.0, 0.0]
    return weights

def stochasticGradAscent(dataMatln, classLabels): #随机梯度上升算法
    dataMatln = np.mat(dataMatln)
    m, n = np.shape(dataMatln)
    alpha = 0.01
    weights = np.ones(n)                          #行向量
    weights = np.mat(weights).transpose()
    for i in range(m):
        h = sigmoid(np.sum(dataMatln[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatln[i]
    return weights

def plotBestFit(w):
    weights = w.getA()
    dataList, labelList = loadDataSet()
    dataArr = np.array(dataList)
    n = len(dataList)
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if labelList[i] == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else :
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s',edgecolors='k')
    ax.scatter(xcord2,ycord2,s=30,c='blue',edgecolors='k')
    x1 = np.arange(-3., 3., 0.1)
    x2 = -1.0*(weights[0][0]+weights[1][0]*x1)/weights[2][0]       #  x2=-1*(b+w1*x)/(w2)
    ax.plot(x1, x2)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


dataList, labelList = loadDataSet()
#b = stochasticGradAscent(dataList, labelList)
a = gradAscent(dataList, labelList)
plotBestFit(a)
#plotBestFit(b)
