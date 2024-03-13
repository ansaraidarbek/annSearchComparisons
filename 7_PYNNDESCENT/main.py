from pynndescent import NNDescent
from time import perf_counter
import numpy as np
import os

def pynndescent_run (name) :
    print("PYNNDESCENT start ----------------------------------------------")
    nameFull = name +'-true-labels.xlsx'
    datasetTrainImages, datasetTestImages, _ = get_ann_benchmark_data2(name)

    def createIndex(indexMethod, datasetImages):
        time_start = perf_counter()
        index = indexMethod(datasetImages, metric="euclidean")
        time_end = perf_counter()
        totalTime = (time_end - time_start)
        print(f'Building time {totalTime:.3f}')
        return (index, totalTime)
    (indexedStruct, time) = createIndex(NNDescent, datasetTrainImages)
    # (min, max) = createIndexNumerous(createIndex, AnnoyIndex, datasetImages, 10)
    # print('min : ', min, '\n','max : ', max,)

    indexes = []
    distances = []
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
        # report the duration
        print(f'Searching time {totalTime:.3f}')
    measureTime(1000, indexes, distances, datasetTestImages)
    
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
    print("PYNNDESCENT end ----------------------------------------------")