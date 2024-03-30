import ngtpy
from time import perf_counter
import numpy as np
import os

def scipy_run (name, metric, runs, queries) :
    print("SCIPY start ----------------------------------------------")
    nameFull = name +'-true-labels.xlsx'
    nameFull = name + '-' + metric + '-true-labels.xlsx'
    datasetTrainImages, datasetTestImages, _ = get_ann_benchmark_data(name)

    def createIndex(indexMethod, datasetImages):
        f = datasetImages.shape[1] # Length of item vector that will be indexed
        indexMethod.create(b"tmp", f)
        index = ngtpy.Index(b"tmp")
        time_start = perf_counter()
        index.batch_insert(datasetImages)
        index.save()
        time_end = perf_counter()
        totalTime = (time_end - time_start)
        return (index, totalTime)
    
    (minBuildTime, maxBuildTime, indexedStruct) = createIndexNumerous(createIndex, ngtpy, datasetTrainImages, runs)

    def measureTime(par, indexes, distances, datasetImages):
        totalTime = 0
        for i in range(par) : 
            xq = datasetImages[i:i+1].astype('float32') # Use the first image as the query vector
            xq = xq.flatten().tolist()
            time_start = perf_counter()
            storage = indexedStruct.search(xq, 100)
            time_end = perf_counter()
            totalTime += (time_end - time_start)
            index = []
            distance = []
            for i in range(len(storage)):
                index.append(storage[i][0])
                distance.append(storage[i][1])
            indexes.append(index[:100])
            distances.append(distance[:100])
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
    print("SCIPY end ----------------------------------------------")
    return [[minBuildTime, maxBuildTime], [minSearchTime, maxSearchTime], R_0, R_01, R_02]