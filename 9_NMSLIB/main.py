import nmslib
from time import perf_counter
import numpy as np
import os

def nmslib_run (name) :
    print("NMSLIB start ----------------------------------------------")
    nameFull = name +'-true-labels.xlsx'
    datasetTrainImages, datasetTestImages, _ = get_ann_benchmark_data2(name)

    def createIndex(indexMethod, datasetImages):
        f = datasetImages.shape[1] # Length of item vector that will be indexed
        index = indexMethod(method='hnsw', space='l2')
        time_start = perf_counter()
        index.addDataPointBatch(datasetImages)
        index.createIndex({'post': 2}, print_progress=True)
        time_end = perf_counter()
        totalTime = (time_end - time_start)
        print(f'Building time {totalTime:.3f}')
        return (index, totalTime)
    (indexedStruct, time) = createIndex(nmslib.init, datasetTrainImages)

    # (min, max) = createIndexNumerous(createIndex, AnnoyIndex, datasetImages, 10)
    # print('min : ', min, '\n','max : ', max,)

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
        # report the duration
        print(f'Searching time {totalTime:.3f}')
        return np.round(totalTime, 3)
    numberOfQueries = 1000
    measureTime(numberOfQueries, indexes, distances, datasetTestImages)
    
    # (min, max) = measureTimeNumerous(measureTime, 10)
    # print('min : ', min, '\n','max : ', max,)

    indexes = np.array(indexes)
    distances = np.round(np.array(distances).astype(float), 4)

    print('indexes : ', indexes.shape)
    print('distances : ', distances.shape)

    fullPath = os.path.dirname(os.path.abspath(__file__))
    path = fullPath + '/datasets/'+nameFull
    (trueIndexes, trueDistances) = readDB(path)

    # compareFirstTen(indexes, distances, trueIndexes, trueDistances)

    calculateRecallAverage(indexes, distances, trueIndexes, trueDistances)
    calculateRecallAverage(indexes, distances, trueIndexes, trueDistances, 1.01)
    calculateRecallAverage(indexes, distances, trueIndexes, trueDistances, 1.1)
    print("NMSLIB end ----------------------------------------------")