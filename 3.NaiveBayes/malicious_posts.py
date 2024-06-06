import numpy as np
import re
import random
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])   # empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)  # empty list containing 0's
    for word in inputSet:
        if word in vocabList:  # if word is in vocabulary list, set output vector to 1
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word: %s in not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # Initialize probabilities
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # Vector addition
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # Element-wise division and log transformation
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

# Load data
data, label = loadDataSet()
print(data)
print(label)

# Create vocabulary list
vocabList = createVocabList(data)
print(vocabList)

# Create word vectors for all documents
trainMatrix = [setOfWords2Vec(vocabList, doc) for doc in data]
print(trainMatrix)

# Train the Naive Bayes model
p0Vect, p1Vect, pAbusive = trainNB0(trainMatrix, label)
print(p0Vect, p1Vect, pAbusive)

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V, p1V, pAB = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry," classified as: ",classifyNB(thisDoc,p0V,p1V,pAB))
    testEntry = np.array(setOfWords2Vec(myVocabList,testEntry))
    print (testEntry, "classifies as: ", classifyNB(thisDoc,p0V,p1V,pAB))
testingNB()