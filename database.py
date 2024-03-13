import pandas as pd
import numpy as np
import os
from keras.datasets import mnist
from urllib.request import urlretrieve
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
    df = pd.read_excel(path)
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

def get_ann_benchmark_data(dataset_name):
    if not os.path.exists(f"../datasets/{dataset_name}.hdf5"):
        print(f"Dataset {dataset_name} is not cached; downloading now ...")
        urlretrieve(f"http://ann-benchmarks.com/{dataset_name}.hdf5", f"../datasets/{dataset_name}.hdf5")
    hdf5_file = h5py.File(f"../datasets/{dataset_name}.hdf5", "r")
    print('trainDataset : ', np.array(hdf5_file['train']).shape)
    print('testDataset : ', np.array(hdf5_file['test']).shape)
    return np.array(hdf5_file['train']).astype('float32'), np.array(hdf5_file['test']).astype('float32'), hdf5_file.attrs['distance']

def get_ann_benchmark_data2(dataset_name):
    fullPath = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(f"{fullPath}/datasets/{dataset_name}.hdf5"):
        print(f"Dataset {dataset_name} is not cached; downloading now ...")
        urlretrieve(f"http://ann-benchmarks.com/{dataset_name}.hdf5", f"{fullPath}/datasets/{dataset_name}.hdf5")
    hdf5_file = h5py.File(f"{fullPath}/datasets/{dataset_name}.hdf5", "r")
    print('trainDataset : ', np.array(hdf5_file['train']).shape)
    print('testDataset : ', np.array(hdf5_file['test']).shape)
    return np.array(hdf5_file['train']).astype('float32'), np.array(hdf5_file['test']).astype('float32'), hdf5_file.attrs['distance']

