
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def compareElems (num, indexes, distances, trueIndexes, trueDistances, epsilon = 1):
    def areSame(a, b) :
        return (a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1])
    if (not areSame(indexes, trueIndexes)):
        return
    for i in range(indexes.shape[0]) :
        for j in range(num) :
            print(str(indexes[i][j]) + " || " + str(trueIndexes[i][j]))
            print(str(distances[i][j]) + " || " + str(trueDistances[i][j]))
        break

def calculateRecallAverage (indexes, distances, trueIndexes, trueDistances, epsilon = 1, show = False):
    def areSame(a, b) :
        return (a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1])
    if (not areSame(indexes, trueIndexes)):
        return
    sums = []
    average = 0
    for i in range(indexes.shape[0]) :
        sum = 0
        for j in range(indexes.shape[1]) :
            # print(str(indexes[i][j]) + " || " + str(trueIndexes[i][j]))
            # print(str(distances[i][j]) + " || " + str(trueDistances[i][j]))
            if (distances[i][j] <= epsilon * trueDistances[i][j]):
                # print(str(distances[i][j]) + " || " + str(trueDistances[i][j]))
                sum+=1
        sum /= indexes.shape[1]
        average += sum
        sums.append(sum)
    average /= indexes.shape[0]
    if (not show):
        msg = 'Recall@' + str(epsilon) + ': ' + str(np.round(average, 4))
        print(msg)
    return np.round(average, 4)

def calculateRecallTotal (indexes, distances, trueIndexes, trueDistances, epsilon = 1, show = False):
    def areSame(a, b) :
        return (a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1])
    if (not areSame(indexes, trueIndexes)):
        return
    sum = 0
    for i in range(indexes.shape[0]) :
        for j in range(indexes.shape[1]) :
            # print(str(indexes[i][j]) + " || " + str(indexes[i][j]))
            # print(str(distances[i][j]) + " || " + str(trueDistances[i][j]))
            if (distances[i][j] <= epsilon * trueDistances[i][j]):
                # print(str(distances[i][j]) + " || " + str(trueDistances[i][j]))
                sum+=1
    sum /= (indexes.shape[0] * indexes.shape[1])
    if (not show):
        print(sum)
    return np.round(sum, 4)

def draw_mnist(indexes, distances, datasetImages):
    arr = np.empty([0,datasetImages.shape[1]])
    for i in range(10):
        print('index : ', indexes[0][i], '\ndistance : ', distances[0][i])
        arr = np.vstack((arr, datasetImages[indexes[0][i]]))
    plt.figure(figsize=(len(arr)*4,4))
    for index, (image) in enumerate(zip(arr)):
        plt.subplot(1, len(arr), index + 1)
        plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)

def maximum (a , b) :
    if a >= b :
        return a
    else :
        return b

def minimum (a , b) :
    if a <= b :
        return a
    else :
        return b

def maxNone (a , b) :
    statement1 = a is None
    statement2 = b is None
    if statement1 and not statement2:
        return b
    if not statement1 and not statement2:
        num = maximum(a, b)
        return num
    if not statement1 and statement2:
        return a
    if statement2 and statement1:
        return None
    
def minNone (a , b) :
    statement1 = a is None
    statement2 = b is None
    if statement1 and not statement2:
        return b
    if not statement1 and not statement2:
        num = minimum(a, b)
        return num
    if not statement1 and statement2:
        return a
    if statement2 and statement1:
        return None
    
def pad(index, k) :
    while (len(index) < k) :
        index.append(-1)

def formatIndexesAndDistance(indexes, distances, k):
    for i in range (len(indexes)):
        if isinstance(indexes[i], np.ndarray):
            indexes[i] = indexes[i].tolist()
        if (len(indexes[i]) > k):
            indexes[i] = indexes[i][:k]
        elif(len(indexes[i]) < k):
            pad(indexes[i], k)
    for i in range (len(distances)):
        if isinstance(distances[i], np.ndarray):
            distances[i] = distances[i].tolist()
        if (len(distances[i]) > k):
            distances[i] = distances[i][:k]
        elif(len(distances[i]) < k):
            pad(distances[i], k)
        
def measureTimeNumerous(function, runs, queries, dataset) :
    queriesNumber = queries
    min = None
    max = None
    indexes = []
    distances = []
    for i in range(runs):
        del indexes[:]
        del distances[:]
        time = float(function(queriesNumber, indexes, distances, dataset))
        print('search ', i + 1, ' done')
        max = maxNone(max, time)
        min = minNone(min, time)
    formatIndexesAndDistance(indexes, distances, 100)
    return (np.round(min, 3), np.round(max, 3), indexes, distances)

def createIndexNumerous(function, indexingMethod, dataset, runs) :
    min = None
    max = None
    indexedStruct = None
    for i in range(runs):
        (indexedStruct, time) = function(indexingMethod, dataset)
        print('index ', i + 1, ' created')
        max = maxNone(max, time)
        min = minNone(min, time)
    return (np.round(min, 3), np.round(max, 3), indexedStruct)

def calculateNormRecall(indexes, trueIndexes, show = False):
    average = 0
    for i in range(indexes.shape[0]):
        for j in range(indexes.shape[1]):
            if (indexes[i][j] in trueIndexes[i]):
                average+=1 
    average/=(indexes.shape[0] * indexes.shape[1])
    if (not show):
        msg = 'norm Recall is ' + str(np.round(average, 4))
        print(msg)
    return np.round(average, 4)


