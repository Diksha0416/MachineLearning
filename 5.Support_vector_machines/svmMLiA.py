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
print(dataArr)
print(labelArr)

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))  # double brackets create 2D array
    iter = 0  # will calculate how many times you have gone through the dataset without changing alpha
    while(iter < maxIter):
        alphasPairsChanges = 0
        # Enter optimization if alphas can be changed.
        for i in range(m):
            # prediction of class
            fXi = float(np.multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
    
            # Both positive and negative margins are tested.
            if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # Randomly select second alpha
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])  # if error is large then the alpha corresponding to this data instance can be optimized
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # Guarantee alphas stay between 0 and C.
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, alphas[j] - alphas[i] + C)
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    continue

                # OPTIMAL AMOUNT TO CHANGE ALPHA J
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
                # Update i by same amount as j in opposite direction
                if eta >= 0:
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    continue

                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])

                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphasPairsChanges += 1
            print("iter: %d i: %d, pairs changes %d" % (iter, i, alphasPairsChanges))
        if alphasPairsChanges == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number:", iter)
    return b, alphas
b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001,5)
print(b,alphas)
for i in range(100):
    if alphas[i] > 0.0:
        print (dataArr[i], labelArr[i])

