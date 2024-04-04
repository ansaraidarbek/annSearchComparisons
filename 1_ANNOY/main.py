from annoy import AnnoyIndex
from time import perf_counter
import numpy as np
import os

def annoy_run (name, metric, runs, queries) :
    print("Annoy start ----------------------------------------------")
    nameFull = name + '-' + metric + '-true-labels.xlsx'
    datasetTrainImages, datasetTestImages, _ = get_ann_benchmark_data(name)

    def createIndex(indexMethod, datasetImages):
        f = datasetImages.shape[1] # Length of item vector that will be indexed
        t = indexMethod(f, 'euclidean')
        time_start = perf_counter()
        for i in range(datasetImages.shape[0]):
            t.add_item(i, datasetImages[i])
        t.build(10) # 10 trees
        time_end = perf_counter()
        totalTime = (time_end - time_start)
        return (t, totalTime)
    
    (minBuildTime, maxBuildTime, indexedStruct) = createIndexNumerous(createIndex, AnnoyIndex, datasetTrainImages, runs)

    indexName = name + '-' + metric + '-index.ann'
    indexedStruct.save(indexName)
    

    indexName = name + '-' + metric + '-index.ann'
    u = AnnoyIndex(datasetTrainImages.shape[1], 'euclidean')
    u.load(indexName) # super fast, will just mmap the file

    def measureTime(par, indexes, distances, datasetTestImages):
        totalTime = 0
        for i in range(par) : 
            time_start = perf_counter()
            (index, distance) = u.get_nns_by_vector(datasetTestImages[i], 100, include_distances=True)
            time_end = perf_counter()
            totalTime += (time_end - time_start)
            indexes.append(index)
            distances.append(distance)
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
    print("Annoy end ----------------------------------------------")
    return [[minBuildTime, maxBuildTime], [minSearchTime, maxSearchTime], minBuildTime + minSearchTime, R_0, R_01, R_02, R_norm]