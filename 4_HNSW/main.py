import hnswlib
from time import perf_counter
import numpy as np
import os

def hnsw_run (name, metric, runs, queries) :
    print("HNSW start ----------------------------------------------")
    nameFull = name +'-true-labels.xlsx'
    nameFull = name + '-' + metric + '-true-labels.xlsx'
    datasetTrainImages, datasetTestImages, _ = get_ann_benchmark_data(name)

    def createIndex(indexMethod, datasetImages):
        d = datasetImages.shape[1] 
        num_elements = datasetImages.shape[0] 
        time_start = perf_counter()
        index = indexMethod(space = 'l2', dim = d)
        index.init_index(max_elements = num_elements, ef_construction = 200, M = 16)
        ids = np.arange(num_elements)
        index.add_items(datasetImages, ids)
        index.set_ef(120)
        time_end = perf_counter()
        totalTime = (time_end - time_start)
        return (index, totalTime)

    (minBuildTime, maxBuildTime, indexedStruct) = createIndexNumerous(createIndex, hnswlib.Index, datasetTrainImages, runs)

    def measureTime(par, indexes, distances, datasetImages):
        totalTime = 0
        for i in range(par) : 
            xq = datasetImages[i:i+1].astype('float32') # Use the first image as the query vector
            time_start = perf_counter()
            index, distance = indexedStruct.knn_query(xq, k = 100)
            time_end = perf_counter()
            totalTime += (time_end - time_start)
            indexes.append(index[0])
            distances.append(np.sqrt(distance[0]))
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