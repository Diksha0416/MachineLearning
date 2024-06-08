import numpy as np
import matplotlib.pyplot as plt
# Set the minError to +infinity
# For every feature in the dataset:
#    For every step:
#       Build a decision stump and test it with the weighted dataset
#       If the error is less than minError: set this stump as the best stump
# Return the best stump
def loadData():
    dataMat = np.matrix([[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels
dataMat , classLabels = loadData()

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
D = np.mat(np.ones((5,1))/5)
x,y,z=buildStump(dataMat,classLabels,D)
print(x,y,z)

# Algorithm
# For each iteration:
#    Find the best stump using buildStump()
#    Add the best stump to the stump array
#    Calculate alpha
#    Calculate the new weight vector -D
#    Update the aggregate class estimate
#    If the error rate == 0.0: break out of the for loop

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m) # holds the weight of each piece of data
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt): # will increase weight of miscassified data and decrease weight of classified data
        bestStump, error, classEst = buildStump(dataArr, classLabels, D) # created decision stump
        print ("D:",D.T)
        alpha = float(0.5*np.log((1.0 - error)/max(error,1e-16))) # will tell the total classifier how much to weight the output from this stump
        bestStump['alpha'] = alpha 
        weakClassArr.append(bestStump)
        print("ClassEst: ", classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum() # calculate new weights D for new ietaration
        aggClassEst += alpha * classEst
        print ("AggClassEst: ",aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate,"\n")
        if errorRate == 0.0:
            break
    return weakClassArr
classifierArray = adaBoostTrainDS(dataMat, classLabels, 9)
print(classifierArray)


def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range (len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print (aggClassEst)
    return np.sign(aggClassEst)
classify = adaClassify([0,0],classifierArray)
print(classify)


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


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