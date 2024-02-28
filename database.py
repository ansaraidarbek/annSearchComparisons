import pandas as pd
import numpy as np
from keras.datasets import mnist

def readDB () :
    df = pd.read_excel('../mnistTrueLabels.xlsx')
    trueIndexes = df['Index'].to_numpy()
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

def readMnist () :
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    datasetImages = np.concatenate((train_X, test_X), axis=0)
    datasetLabels = np.concatenate((train_y, test_y), axis=0)
    datasetImages = datasetImages.reshape(datasetImages.shape[0], datasetImages.shape[1] * datasetImages.shape[2])
    datasetImages = datasetImages.astype('float32') 
    print('datasetImages : ', datasetImages.shape)
    print('datasetLabels : ', datasetLabels.shape)
    return (datasetImages, datasetLabels)

