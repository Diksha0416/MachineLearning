import numpy as np
import operator
import matplotlib.pyplot as plt

# K-Nearest Neighbors classifier function
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # Calculate the difference matrix
    sqDiffMat = diffMat ** 2  # Square the differences
    sqDistances = sqDiffMat.sum(axis=1)  # Sum the squared differences
    distances = sqDistances ** 0.5  # Take the square root to get the Euclidean distances
    sortedDisIndices = distances.argsort()  # Sort distances and get the sorted indices
    classCount = {}  # Dictionary to count the occurrences of each label
    for i in range(k):
        voteIlabel = labels[sortedDisIndices[i]]  # Get the label of the i-th nearest neighbor
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # Count the label
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # Sort by count
    return sortedClassCount[0][0]  # Return the label with the highest count

# Function to load dataset from a file
def file2matrix(filename):
    with open(filename) as fr:
        # Read all lines from the file
        lines = fr.readlines()
    
    numberOfLines = len(lines)  # Number of lines in the file
    returnMat = np.zeros((numberOfLines, 3))  # Initialize the matrix
    classLabelVector = []  # Initialize the class label vector

    for index, line in enumerate(lines):
        line = line.strip()  # Remove leading and trailing whitespace
        listFromLine = line.split('\t')  # Split the line by tab
        returnMat[index, :] = listFromLine[0:3]  # First 3 columns are features
        classLabelVector.append(listFromLine[-1])  # Last column is the label

    return returnMat, classLabelVector  # Return the matrix and label vector
datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
print("Dating Data",datingDataMat)
print("Dating labels",datingLabels)
# normalizing code
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # Compute the minimum value for each feature
    maxVals = dataSet.max(0)  # Compute the maximum value for each feature
    ranges = maxVals - minVals  # Compute the range for each feature

    m = dataSet.shape[0]  # Number of data points
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # Subtract the minimum value
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # Divide by the range

    return normDataSet, ranges, minVals  # Return the normalized data, ranges, and min values

normDataSet, ranges, minVals = autoNorm(datingDataMat)
print("NormDataSets",normDataSet)
print("Ranges",ranges)
print("Min values",minVals)

# classifier testing code for dating site
def datingClassTest():
    horatio = 0.10# Hold-out ratio: proportion of the dataset to be used as the test set

    m = normDataSet.shape[0] # Determine the no of data points
    numTestVecs = int(m*horatio) # Calculate the number of text vectors

    errorCount = 0.0 #initialize errorcount
    for i in range(numTestVecs):
        # Classify the i-th test vector using the KNN algorithm
        classifierResult = classify0(normDataSet[i,:], normDataSet[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %s, the real answer is: %s"%(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("The total error rate is: %f"(errorCount/float(numTestVecs)))

#datingClassTest()

def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']

    # Collect input from the user    
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    # Create the input array and normalize it
    inArr = np.array([ffMiles,percentTats,iceCream])

    # Classify the input
    classifierResult = classify0((inArr-minVals)/ranges,normDataSet,datingLabels,3)
    # Print the Result
    print ("You will probably like this person: ",resultList[int(classifierResult)-1])
#Example usage
classifyPerson()