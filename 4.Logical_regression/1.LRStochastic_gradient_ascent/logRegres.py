import math as m
import numpy as np
import random
import matplotlib.pyplot as plt
def loadDataSet():
    dataMat = []
    labelMat = []
    with open('testSet.txt') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            if len(lineArr) < 3:
                print(f"Skipping line: {line}")
                continue
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            # first column(bias term) accounts for cases where the decision boundary does not pass through the origin in feature space
            # second column is first feature(first column of file)
            # third column is second feature (second column of file)
            labelMat.append(int(lineArr[2]))
            # label used last column of file
    return dataMat, labelMat
x,y = loadDataSet()

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent (dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn) # generating matrix of dataMAt
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix) # tells no of rows and column in matrix
    aplha = 0.001
    maxCycles = 500
    weights = np.ones((n,1)) # means that n rows will conatin 1.0 as a element
    print(weights)
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + aplha * dataMatrix.transpose()* error
    return weights
weights = gradAscent(x,y)
#calculating the error between the actual class 
#and the predicted class and then moving in the direction of that error.
print(weights)

def plotBestFit(wei):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]

    # Separate data points based on class labels
    xcord1 = []; ycord1 =[]
    xcord2 = []; ycord2 =[]
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])

    #plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    #plot the decision boundary
    x= np.arange(-3.0,3.0,1.0) # range of x values for decision boundary
    
    wei = np.array(wei).flatten()
    
    y = (-wei[0] - wei[1] * x) / wei[2] # calculate corresponding y values

    
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

plotBestFit(weights)

# Stochastic gradient Ascent
def stocGradAscent0(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights
weight2 = stocGradAscent0(np.array(x),y)
print(weight2)
plotBestFit(weight2)


# Modified stochastic gradient ascent (20 passes)
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        for i in range(m):
            dataIndex=list(range(m))
            alpha = 4/(1.0+j+1)+0.01 #changes with each iteration :- will improve the osicllation that occur in the dataset
            # You need to do this so that after a large no of cycles, new data still has some impact
            # Alpha isn't strictly decreaing when j<<max(i)
            # j:- no of time you are moving through the dataset
            randIndex = int(random.uniform(0,len(dataIndex))) #update vectors are randomly selected
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
weight3 = stocGradAscent1(np.array(x),y)
print(weight3)
plotBestFit(weight3)

# logistic regression classification function
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0



