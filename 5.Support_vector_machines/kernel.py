import random
import numpy as np
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split(' ')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

dataArr, labelArr = loadDataSet('testSetData.txt')
#print(dataArr)
#print(labelArr)


# Modify selectJ function to use a list of tuples
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = (True, Ei)  # Use tuple (True, Ei) instead of list [True, Ei]
    validEcacheList = [k for k, (isValid, _) in enumerate(oS.eCache) if isValid]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            # Choose j for max step size
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej  # Make sure to return values in all branches of the function


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

class optStruct:
    def __init__(self, dataMainIn, classLabels, C, toler, kTup):
        self.X = dataMainIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMainIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = [(True, 0.0)] * self.m  # Initialize eCache with tuples
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)#kTup contains ifo about kernel

def calcEk(oS,k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i]>0)):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L ==H :
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0: 
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS,i)

        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i,j]-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if(0<oS.alphas[i] and (oS.C > oS.alphas[i])): oS.b = b1
        elif(0<oS.alphas[j]) and (oS.C > oS.alphas[j]):oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0
def kernelTrans(X, A, kTup):
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin': K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K / (-1*kTup[1]**2))
    else: raise NameError('Houston We have a problem that Kernel is not recognized')
    return K
# Outer loop where we select for first alpha
# Here iteration is 1 pass..... It dosen't means no alpha change
# It will stop if there are any oscillations(superior than before) 
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin',0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(),C,toler,kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        # Go over all values
        if entireSet:
            for i in range(oS.m):
                # calling inner loop for second alpha
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i: %d, pairs changed %d" %(iter,i,alphaPairsChanged))
            iter += 1
        else:
        # Go over non bound values
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ("Non-bound, iter: %d i: %d, pairs changed %d" % (iter,i,alphaPairsChanged))
                iter += 1
                # toggle between non bound pass and full pass loop and print out iteration number
        if entireSet: entireSet = False
        elif (alphaPairsChanged == 0): entireSet = True
        print ("iteration number: %d" % iter)
    return oS.b, oS.alphas

b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 5,('rbf',1.3))
#print(b,alphas)

def calcWs(alphas,dataArr,classLabels):
    X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w
#print(calcWs(alphas,dataArr,labelArr))


def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("There are %d support vectors" % np.shape(sVs)[0])
    m,n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):errorCount += 1
    print ("The training error rate is : %f "%(float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRbF2.txt')
    m,n = np.shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!= np.sign(labelArr[i]) : errorCount += 1
    print ("The test error rate is : %f" %(float(errorCount))/m)

testRbf()