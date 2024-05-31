from numpy import array, tile
import operator  # used in knn module for sorting

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # distance calculation
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    # sum of all the distances in the row
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # arranging the distance in ascending order
    sortedDistIndices = distances.argsort()
    # initialize an empty dictionary to count the occurrences of each class among the k nearest neighbours
    classCount = {}

    for i in range(k):
        # voting with lowest k distances
        # with label we get the label of the ith nearest neighbour
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    
    # sorts the classCount dictionary by the counts in descending order
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # returns the label with the highest count among the k nearest neighbors
    return sortedClassCount[0][0]

# Example usage
group, labels = createDataSet()
result = classify0([0, 0], group, labels, 3)
print("The class of [0, 0] is:", result)