def loadDataSet():
    return [[1,3,4],
            [2,3,5],
            [1,2,3,5],
            [2,5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return C1

def judge_is_contain(a,b) -> bool:#判断 a 是否是 b 的子集
    a = set(a)
    b = set(b)
    if a.issubset(b):
        return True
    else:
        return False

def scanD(D, Ck, minSupport): #python3这里需要使用迭代器
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if judge_is_contain(can,tid):
                temp = tuple(can)
                if temp not in ssCnt:
                    ssCnt[temp] = 1
                else:
                    ssCnt[temp] += 1
    numItems = float(len(D))
    retList = []                        #过滤list
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.append(key)
        supportData[key] = support
    return retList, supportData
    # ToDo:这里的retList是这种[(1,), (3,), (2,), (5,)]
    # ToDo:supportData = {(1,): 0.5, (3,): 0.75, (4,): 0.25, (2,): 0.75, (5,): 0.75} 是set类型

def aprioriGen(Lk, k):  #create Ck
    retList = []        #传参数的时候要注意数组先变为set
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort();L2.sort()
            if L1 == L2:
                temp = []
                temp.extend(Lk[i])
                temp.extend(Lk[j])
                retList.append(tuple(set(temp)))
    return retList

def apriori(dataSet, minSupport):
    C1 = createC1(dataSet)             #[[1], [2], [3], [4], [5]]
    D = dataSet                        #[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while(len(L[k-2])>0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supk = scanD(D, Ck, minSupport)
        supportData.update(supk)
        L.append(Lk)
        k += 1
    return L, supportData
#上面的代码仅仅实现了找频繁相机，没有实现关联相机的查找   另外实现了关联项集之后想想 好玩的例子看看knn
#TODO:下面从频繁项开始构建关联规则

def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):  # 不要[[3],[2],[5]]这种，要[2,5]这种
        for freqList in L[i]:   # freqList是tuple类型
            H1 = [ [item] for item in freqList]   #H1这里是list
            if (i > 1):
                rulesFromConseq(freqList, H1, supportData, bigRuleList, minConf)  # 进一步合并
            else:
                calConf(freqList, H1, supportData, bigRuleList, minConf)
    return bigRuleList
'''
freqList = 2,3,4,5
H1 = [[2], [3], [4], [5]]
'''

def calConf(freqList, H, supportData, br1, minConf=0.7):
    prunedH = []
    for conseq in H:
        set1 = set(freqList)
        set2 = set(conseq)
        tuple_delta = tuple(set1 - set2)
        conf = supportData[freqList] / supportData[tuple_delta]  #这里相当于用了一个条件概率来计算p->h
        conf2 = supportData[freqList] / supportData[tuple(conseq)]
        if conf >= minConf:
            if [tuple_delta,tuple(conseq)] not in br1:
                print("{} --> {} == {}".format(tuple_delta, tuple(conseq), conf))
                br1.append([tuple_delta,tuple(conseq)])
                prunedH.append(tuple(conseq))
        if conf2 >= minConf:
            if [tuple(conseq),tuple_delta] not in br1:
                print("{} --> {} == {}".format(tuple(conseq), tuple_delta, conf))
                br1.append([tuple(conseq),tuple_delta])
                prunedH.append(tuple_delta)
    return prunedH

def rulesFromConseq(freqList, H, supportData, br1, minConf = 0.7):    #[2,3,5]
    m = len(H[0])
    if len(freqList) > m + 1:
        Hmp1 = aprioriGen(H,m + 1)
        Hmp1 = calConf(freqList, Hmp1, supportData, br1, minConf)
        if len(Hmp1)>1:
            rulesFromConseq(freqList, Hmp1, supportData, br1, minConf)

L, supportData = apriori(loadDataSet(), 0.5)
generateRules(L, supportData, 0.5)
