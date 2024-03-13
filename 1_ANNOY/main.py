from annoy import AnnoyIndex
from time import perf_counter
import numpy as np

def annoy_run (name) :
    print("Annoy start ----------------------------------------------")
    nameFull = name +'-true-labels.xlsx'
    datasetTrainImages, datasetTestImages, _ = get_ann_benchmark_data2(name)

    def createIndex(indexMethod, datasetImages):
        f = datasetImages.shape[1] # Length of item vector that will be indexed
        t = indexMethod(f, 'euclidean')
        time_start = perf_counter()
        for i in range(datasetImages.shape[0]):
            t.add_item(i, datasetImages[i])
        t.build(10) # 10 trees
        time_end = perf_counter()
        totalTime = (time_end - time_start)
        print(f'Building time {totalTime:.3f}')
        return (t, totalTime)
    
    (indexedStruct, time) = createIndex(AnnoyIndex, datasetTrainImages)
    # (min, max) = createIndexNumerous(createIndex, AnnoyIndex, datasetImages, 10)
    # print('min : ', min, '\n','max : ', max,)
    indexName = name + '-index.ann'
    indexedStruct.save(indexName)
    

    indexName = name + '-index.ann'
    u = AnnoyIndex(datasetTrainImages.shape[1], 'euclidean')
    u.load(indexName) # super fast, will just mmap the file

    indexes = []
    distances = []
    
    def measureTime(par, indexes, distances, datasetTestImages):
        totalTime = 0
        for i in range(par) : 
            time_start = perf_counter()
            (index, distance) = u.get_nns_by_vector(datasetTestImages[i], 100, include_distances=True)
            time_end = perf_counter()
            totalTime += (time_end - time_start)
            indexes.append(index)
            distances.append(distance)
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

    path = './datasets/'+nameFull
    (trueIndexes, trueDistances) = readDB(path)

    # compareFirstTen(indexes, distances, trueIndexes, trueDistances)

    calculateRecallAverage(indexes, distances, trueIndexes, trueDistances)
    calculateRecallAverage(indexes, distances, trueIndexes, trueDistances, 1.01)
    calculateRecallAverage(indexes, distances, trueIndexes, trueDistances, 1.1)
    print("Annoy end ----------------------------------------------")