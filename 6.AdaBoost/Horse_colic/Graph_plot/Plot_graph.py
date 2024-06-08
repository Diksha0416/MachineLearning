import matplotlib.pyplot as plt
import numpy as np
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split(' '))
    print(numFeat)
    dataMat = []
    labelMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            curLine = line.strip().split(' ')
            # Check if the current line has the expected number of features
            if len(curLine) != numFeat:
                continue
            try:
                lineArr = [float(curLine[i]) for i in range(numFeat - 1)]
                labelMat.append(float(curLine[-1]))
                dataMat.append(lineArr)
            except ValueError as e:
                continue
    return dataMat, labelMat


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))# first all are done 1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0 #values that dont meet the inequality are thrown to -1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = 1.0 # rest remain 1
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T # first checked input data is in proper format 
    m,n = np.shape(dataMatrix)
    numSteps = 10.0 # used to iterate over all the possible values 
    bestStump = {} # empty dictionary which is used store classifier info acc to best choice of a decision stump given this weight vector D
    bestClasEst = np.mat(np.zeros((m,1)))
    minError = np.inf # used to find min possible error rate
    for i in range(n): # goes over all features of our dataset
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range (-1,int(numSteps)+1):
            for inequal in ['lt', 'gt']: # toggles your inequality b/w > and <
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                # calculated weighted error
                print ("Split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i,threshVal,inequal,weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)  # Initialize weights to 1/m
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        if error == 0.0:  # If error is 0, break the loop
            break
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # Calculate alpha
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()  # Normalize weights to sum to 1
        print(f"Iteration {i+1}:")
        print(f"Weights D: {D.T}")
        aggClassEst += alpha * classEst
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        if errorRate == 0.0:  # If error rate is 0, break the loop
            break
    return weakClassArr,aggClassEst



def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range (len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print (aggClassEst)
    return np.sign(aggClassEst)



def plotROC(predStrengths, classLabels):
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(np.array(classLabels) == 1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX], [cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print("The Area Under the Curve is : ", ySum*xStep)

dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
classifierArray,aggClassEst = adaBoostTrainDS(dataArr,labelArr,10)
plotROC(aggClassEst.T,labelArr)
