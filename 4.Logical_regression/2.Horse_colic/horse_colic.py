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

    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + aplha * dataMatrix.transpose()* error
    return weights
weights = gradAscent(x,y)

# Modified stochastic gradient ascent (20 passes)
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m, n = dataMatrix.shape if len(dataMatrix.shape) > 1 else (len(dataMatrix), 1)
    
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

# logistic regression classification function
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0


def colicTest():
    frTrain = open("horse-colic.data")
    frTest = open('horse-colic.test')
    trainingSet = []
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine = line.strip().split()
#       print("Current line:", currLine)
#        print(len(currLine))
        if len(currLine) < 22 or '?' in currLine:  # Check if the line has enough elements
            continue  # Skip lines with insufficient elements
        lineArr =[]
        for i in range(21):
            try:
                lineArr.append(float(currLine[i]))
            except ValueError:
                lineArr.append(random.uniform(0.0, 100.0))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

#    if trainingSet:
#       print("First few elements of trainingSet:")
#        for i in range(min(5, len(trainingSet))):  # Print at most 5 elements
#            print(trainingSet[i])  
#    else:
#        print("Training set is empty.")   

    trainingWeights = stocGradAscent1(np.array(trainingSet), trainingLabels,500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split()
        lineArr = []
        if len(currLine) < 22 or '?' in currLine:  # Check if the line has enough elements
            continue  # Skip lines with insufficient elements
        for i in range(21):
            try:
                lineArr.append(float(currLine[i]))
            except ValueError:
                lineArr.append(random.uniform(0.0, 100.0))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
        if float(classifyVector(np.array(lineArr),trainingWeights))!= float(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("The error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        error = colicTest()
        errorSum += error
    print("After %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

multiTest()