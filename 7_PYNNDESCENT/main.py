from pynndescent import NNDescent
from time import perf_counter
import numpy as np
import os

def pynndescent_run (name, metric, runs, queries) :
    print("PYNNDESCENT start ----------------------------------------------")
    nameFull = name +'-true-labels.xlsx'
    nameFull = name + '-' + metric + '-true-labels.xlsx'
    datasetTrainImages, datasetTestImages, _ = get_ann_benchmark_data(name)

    def createIndex(indexMethod, datasetImages):
        time_start = perf_counter()
        index = indexMethod(datasetImages, metric="euclidean")
        time_end = perf_counter()
        totalTime = (time_end - time_start)
        return (index, totalTime)

    (minBuildTime, maxBuildTime, indexedStruct) = createIndexNumerous(createIndex, NNDescent, datasetTrainImages, runs)

    def measureTime(par, indexes, distances, datasetImages):
        totalTime = 0
        for i in range(par) : 
            xq = datasetImages[i:i+1].astype('float32') # Use the first image as the query vector
            time_start = perf_counter()
            index, distance = indexedStruct.query(xq, k=100)
            time_end = perf_counter()
            totalTime += (time_end - time_start)
            indexes.append(index[0])
            distances.append(distance[0])
            # distances.append(np.sqrt(distance[0]))
            # indexes.append(index[0])
        return np.round(totalTime, 3)

    (minSearchTime, maxSearchTime, indexes, distances) = measureTimeNumerous(measureTime, runs, queries, datasetTestImages)

    indexes = np.array(indexes)
    distances = np.round(np.array(distances).astype(float), 4)

    print('indexes : ', indexes.shape)
    print('distances : ', distances.shape)

    fullPath = os.path.dirname(os.path.abspath(__file__))
    path = fullPath + '/datasets/'+nameFull
    (trueIndexes, trueDistances) = readDB(path)

    # amount = 10
    # compareElems(amount, indexes, distances, trueIndexes, trueDistances)

    R_0 = calculateRecallAverage(indexes, distances, trueIndexes, trueDistances, 1, True)
    R_01 = calculateRecallAverage(indexes, distances, trueIndexes, trueDistances, 1.01, True)
    R_02 = calculateRecallAverage(indexes, distances, trueIndexes, trueDistances, 1.1, True)
    R_norm = calculateNormRecall(indexes, trueIndexes, True)
    print("PYNNDESCENT end ----------------------------------------------")
    return [[minBuildTime, maxBuildTime], [minSearchTime, maxSearchTime], minBuildTime + minSearchTime, R_0, R_01, R_02, R_norm]