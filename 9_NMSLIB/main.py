import nmslib
from time import perf_counter
import numpy as np
import os

def nmslib_run (name, metric, runs, queries) :
    print("NMSLIB start ----------------------------------------------")
    nameFull = name +'-true-labels.xlsx'
    nameFull = name + '-' + metric + '-true-labels.xlsx'
    datasetTrainImages, datasetTestImages, _ = get_ann_benchmark_data(name)

    def createIndex(indexMethod, datasetImages):
        f = datasetImages.shape[1] # Length of item vector that will be indexed
        index = indexMethod(method='hnsw', space='l2')
        time_start = perf_counter()
        index.addDataPointBatch(datasetImages)
        index.createIndex({'post': 2}, print_progress=True)
        time_end = perf_counter()
        totalTime = (time_end - time_start)
        return (index, totalTime)

    (minBuildTime, maxBuildTime, indexedStruct) = createIndexNumerous(createIndex, nmslib.init, datasetTrainImages, runs)

    indexes = []
    distances = []
    def measureTime(par, indexes, distances, datasetTestImages):
        totalTime = 0
        for i in range(par) : 
            time_start = perf_counter()
            index, distance = indexedStruct.knnQuery(datasetTestImages[i], k=100)
            time_end = perf_counter()
            totalTime += (time_end - time_start)
            indexes.append(index)
            distances.append(np.sqrt(distance))
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
    print("HNSW end ----------------------------------------------")
    return [[minBuildTime, maxBuildTime], [minSearchTime, maxSearchTime], R_0, R_01, R_02]