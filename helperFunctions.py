import numpy as np
import matplotlib.pyplot as plt
def compareFirstTen (indexes, distances, trueIndexes, trueDistances, epsilon = 1):
    def areSame(a, b) :
        return (a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1])
    if (not areSame(indexes, trueIndexes)):
        return
    for i in range(indexes.shape[0]) :
        for j in range(10) :
            print(str(indexes[i][j]) + " || " + str(trueIndexes[i][j]))
            print(str(distances[i][j]) + " || " + str(trueDistances[i][j]))
        break

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

def calculateRecallAverage (indexes, distances, trueIndexes, trueDistances, epsilon = 1):
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
    print(f"Recall@{epsilon}: {average:.4f}")

def calculateRecallTotal (indexes, distances, trueIndexes, trueDistances, epsilon = 1):
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
    print(sum)

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
    
        
def measureTimeNumerous(function, n) :
    queriesNumber = 1000
    indexes = []
    distances = []
    min = None
    max = None
    for i in range(n):
        time = float(function(queriesNumber, indexes, distances))
        max = maxNone(max, time)
        min = minNone(min, time)
    return (min, max)

def createIndexNumerous(function, indexingMethod, dataset, n) :
    min = None
    max = None
    for i in range(n):
        (_, time) = function(indexingMethod, dataset)
        max = maxNone(max, time)
        min = minNone(min, time)
    return (min, max)

