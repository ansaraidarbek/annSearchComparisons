import faiss
from time import perf_counter
import numpy as np
import os



def faiss_run (name, metric, runs, queries, method) :
    print("FAISS start ----------------------------------------------")
    nameFull = name +'-true-labels.xlsx'
    nameFull = name + '-' + metric + '-true-labels.xlsx'
    datasetTrainImages, datasetTestImages, _ = get_ann_benchmark_data(name)
    def createIndex(indexMethod, datasetImages):
        d = datasetImages.shape[1] # dimension
        if (method == 'hnsw'):
            M = 16
            time_start = perf_counter()
            index = indexMethod(d, M)
            index.add(datasetImages) 
            time_end = perf_counter()
            totalTime = (time_end - time_start)
            return (index, totalTime)
        M = 100
        time_start = perf_counter()
        index = indexMethod(faiss.IndexFlatL2(d), d, M, faiss.METRIC_L2)
        index.train(datasetImages)
        index.add(datasetImages) 
        time_end = perf_counter()
        totalTime = (time_end - time_start)
        return (index, totalTime)
    
    index = faiss.IndexHNSWFlat if method == 'hnsw' else faiss.IndexIVFFlat
    (minBuildTime, maxBuildTime, indexedStruct) = createIndexNumerous(createIndex, index, datasetTrainImages, runs)

    def measureTime(par, indexes, distances, datasetImages):

        k=100
        totalTime = 0
        for i in range(par) : 
            xq = datasetImages[i:i+1].astype('float32') # Use the first image as the query vector
            time_start = perf_counter()
            distance, index = indexedStruct.search(xq, k) 
            time_end = perf_counter()
            totalTime += (time_end - time_start)
            distances.append(np.sqrt(distance[0]))
            indexes.append(index[0])
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
    print("FAISS end ----------------------------------------------")
    return [[minBuildTime, maxBuildTime], [minSearchTime, maxSearchTime], R_0, R_01, R_02]