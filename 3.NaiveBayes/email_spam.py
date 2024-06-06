import numpy as np
import re
import random

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


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
# takes a big string and parses out the text into a list of strings
def textParse(bigString):
    listOfTokens = re.split(r'\W+',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2] # to lowercase

def spamTest():
    docList = []
    classList = []
    fullText = []
    # load and parse text files
    for i in range(1,26):
        wordList = textParse(open('spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    # randomly create the training set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        # number is selected removed from trainingSet and moved to testSet
        ##This randomly selecting a portion of our data for the training set and a portion for the test set is called hold-out cross validation.
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # if email isnt classified correctly the error count is incremented
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print("The error rate is : ", float(errorCount)/len(testSet))
spamTest()