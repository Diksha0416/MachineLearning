from numpy import *
import urllib
import json
from time import sleep
import matplotlib
import matplotlib.pyplot as plt
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power((vecA - vecB),2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)
    return centroids

dataMat = mat(loadDataSet('TestSet.txt'))
centroids =randCent(dataMat,2)


###################

def kMeans(dataSet, k, distMeas = distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf 
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j 
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist ** 2
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis = 0)
    return centroids, clusterAssment
myCentriods, clustAssing = kMeans(dataMat,4)


def biKmeans(dataSet, k, disMeas=distEclud):
    m =  shape(dataSet)[0]
    clusterAssment =  mat( zeros((m, 2)))
    centroid0 =  mean(dataSet, axis=0).tolist()
    centList = [centroid0]

    for j in range(m):
        clusterAssment[j, 1] = disMeas( mat(centroid0), dataSet[j, :])**2

    while len(centList) < k:
        lowestSSE = float('inf')
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[ nonzero(clusterAssment[:, 0].A == i)[0], :]
            if len(ptsInCurrCluster) == 0:
                continue
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, disMeas)
            sseSplit =  sum(splitClustAss[:, 1])
            sseNotSplit =  sum(clusterAssment[ nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("SSE split and SSE not split: ", sseSplit, "   ", sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        bestClustAss[ nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[ nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print("the bestCentToSplit is: ", bestCentToSplit)
        print("the len of bestClusterAss is: ", len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[ nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss

    return  mat(centList), clusterAssment
import requests
import pandas as pd
import io
urlData = requests.get('https://api.data.gov.in/resource/bccc6a91-cde0-4d1a-b255-6aab90a9e303?api-key=579b464db66ec23bdd00000104d25d16a7284b96728a24b871ae1380&format=json&limit=1000')

data = urlData.json()
df = pd.DataFrame(data['records'])
print(df)

specific_column=df[['latitude___n','longitude___e']]

print("Specific Columns:")
print(specific_column)
specific_column.insert(0, 'line_number', range(1, len(specific_column) + 1))
specific_column.to_csv('specific_column.txt', sep=' ', index = False, header= False)

def distSLC(vecA, vecB):
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * cos(pi * (vecB[0,0] - vecA[0,0])/180)
    return arccos(a+b)*6371.0

import matplotlib.pyplot as plt
from matplotlib.image import imread

def clusterClubs(numClust):
    # Load data from file
    datList = []
    with open('specific_column.txt') as file:
        for line in file:
            lineArr = line.strip().split()
            datList.append([float(lineArr[1]), float(lineArr[2])])
    datMat = mat(datList)
    
    # Perform bi-directional k-means clustering
    myCentroids, clustAssing = biKmeans(datMat, numClust, disMeas=distSLC)
    
    # Plotting setup
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]  # Adjust as needed
    ax1 = fig.add_axes(rect)
    
    # Plot each cluster
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90)
    
    # Plot centroids
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    
    # Display the plot
    plt.show()
    plt.draw()  # Update the figure with the plotted data

clusterClubs(5)
