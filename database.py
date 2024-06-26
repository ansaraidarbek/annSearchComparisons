import pandas as pd
import numpy as np
import os
from six.moves import urllib
from openpyxl import load_workbook 
import h5py

def exportDB (indexes, distances, name, type) :
    print('indexes : ', indexes.shape)
    distances = np.round(np.array(distances).astype(float), 4)
    print('distances : ', distances.shape)
    df = pd.DataFrame(pd.Series(list(indexes)), columns=['Indexes'])
    df["Distances"] = pd.Series(list(distances))
    out_path = "../datasets/" + name 
    print(df.shape)
    print(df.head)
    if (type == 0) :
        df.to_csv(out_path, index=False) 
    elif (type == 1) :
        df.to_excel(out_path)

def readDB (path) :
    wb = load_workbook(filename = path)
    ws = wb['Sheet1']

    data = ws.values

    # Get the first line in file as a header line

    columns = next(data)[0:]

    # Create a DataFrame based on the second and subsequent lines of data

    df = pd.DataFrame(data, columns=columns)
    trueIndexes = df['Indexes'].to_numpy()
    trueDistances = df['Distances'].to_numpy()
    trueIndexes = convertToNumpyArr(trueIndexes)
    trueDistances = convertToNumpyArr(trueDistances)
    print('trueIndexes : ', trueIndexes.shape)
    print('trueDistances : ', trueDistances.shape)
    return (trueIndexes, trueDistances)

def convertToNumpyArr(column) :
    newColumn = []
    for i in range(len(column)):
        x = column[i].replace("[","")
        x = x.replace("]","")
        x = x.split()
        x = [eval(i) for i in x]
        newColumn.append(x)
    return np.array(newColumn)

# def readMnist () :
#     (train_X, train_y), (test_X, test_y) = mnist.load_data()
#     datasetImages = np.concatenate((train_X, test_X), axis=0)
#     datasetLabels = np.concatenate((train_y, test_y), axis=0)
#     datasetImages = datasetImages.reshape(datasetImages.shape[0], datasetImages.shape[1] * datasetImages.shape[2])
#     datasetImages = datasetImages.astype('float32') 
#     print('datasetImages : ', datasetImages.shape)
#     print('datasetLabels : ', datasetLabels.shape)
#     return (datasetImages, datasetLabels)

def returnAnnBenchmarkName (name) :
    def convertName(name) :
        newName = ''
        for i in name: 
            if (i.isupper()):
                newName += '-'
            newName += i.lower()
        return newName
    metrics = {
    'euclidean' : ['fashionMnist-784', 'gist-960', 'mnist-784', 'sift-128'],
    'angular' : ['deepImage-96', 'glove-25', 'glove-50', 'glove-100', 'glove-200', 'nytimes-256'],
    'dot' : ['lastfm-64'],
    'jaccard' : ['kosarak', 'movielens10m']
    }
    for key in metrics:
        if name in metrics[key]:
            return convertName(name) + '-' + key
    return False

def pathChanger (path) :
    ignorePaths = ['ANNOY', 'SCLEARN', 'MRPT', 'HNSW', 'FAISS', 'DATASKETCH', 'PYNNDESCENT', "SCIPPY", "NMSLIB", "NGT", "FLANN"]
    for i in range(len(ignorePaths)):
        path = path.replace('\\' + str(i+1) + '_' + ignorePaths[i], '')
        path = path.replace('/' + str(i+1) + '_' + ignorePaths[i], '')
    return path

def get_ann_benchmark_data(dataset_name):
    fullPath = None
    try:
        fullPath = os.path.dirname(os.path.abspath(__file__))
    except:
        path = os.getcwd()
        fullPath = pathChanger(path)
    absolutePath = fullPath + '/datasets/' + dataset_name + '.hdf5'
    message = 'Dataset ' + dataset_name + ' is not cached; downloading now ...'
    if not os.path.exists(absolutePath):
        print(message)
        annBenchmarkName = returnAnnBenchmarkName(dataset_name)
        annBenchmarkPath = 'http://ann-benchmarks.com/' + annBenchmarkName + '.hdf5'
        urllib.request.urlretrieve(annBenchmarkPath, absolutePath)
        # urlretrieve(annBenchmarkPath, absolutePath)
    hdf5_file = h5py.File(absolutePath, "r")
    print('trainDataset : ', np.array(hdf5_file['train']).shape)
    print('testDataset : ', np.array(hdf5_file['test']).shape)
    return np.array(hdf5_file['train']).astype('float32'), np.array(hdf5_file['test']).astype('float32'), hdf5_file.attrs['distance']

