from math import log
import operator
import matplotlib.pyplot as plt

def calcShannonEnt (dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

myDat,labels=createDataSet()
#print(myDat)
x=calcShannonEnt(myDat)
print(x)
myDat[0][2]='maybe'
print(myDat)
x=calcShannonEnt(myDat)
print(x)

#measure the entropy
#split the dataset
#measure the entropy on split sets
def splitDataSet(dataSet,axis,value):
    #Dataset we will split
    #ofeature we will split on
    #value of feature to return
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #cut out the feature that you split on 
            reducedFeatVec = featVec[:axis] # we are creating a new list everytime as we dont want to modify the data of actual list
            reducedFeatVec.extend(featVec[axis+1:]) # adds number of list in the current list
            retDataSet.append(reducedFeatVec) # adds a list in a list
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        #loops over all the features of oue dataset
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) # as it contain only unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i   # index of best feature to split on
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # Stop when all classes are equal
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # when you dont meet the stopping condition then this is used to choose the best feature
    if len(dataSet[0]) == 1:
        return majorityCnt (classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeat]
    #mytree dictionary is used to store the tree
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set (featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

mytree = createTree(myDat,labels)
print(mytree) 

decisionNode = dict(boxstyle = "sawtooth", fc="0.8")
leadNode = dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def createPlot():
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decision node', (0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('a leaf node', (0.8,0.1),(0.3,0.8),leadNode)
    plt.show()

#createPlot()

#tells number of leaf nodes to properly size things in X direction
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr =list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

#levels for properly sizing things in Y direction
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

# Plot tree function
# Plots text between child and parent
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    # calculation of width and height of the tree.
    numLeafs = getNumLeafs(myTree)
    getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    #Two global variables are set up to store the width (plotTree.totalW) and depth of the tree (plotTree.totalD).
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff,plotTree.yOff),cntrPt,leadNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree, (0.5,1.0),'')
    plt.show()

createPlot(mytree)

def classify(inputTrree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel

#####################################3
# in place of plotting a tree we can python module 'pickle' foe serializing
def storeTree( putTree, filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)