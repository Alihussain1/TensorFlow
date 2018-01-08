import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import keras
import tensorflow as tf

def one_hot_encode_labels(y_data,num_classes=10):
    numOfLabels = y_data.shape[0] #How Many rows in data ?
    indexOffset = np.arange(numOfLabels) * num_classes #Returns array of [0,10,20,....,numOfLabels]
    yDataOneHotEncoded = np.zeros((numOfLabels,num_classes)) #array of shape (numOfLabels,numClasses)=>e.x:(50000,10)
    yDataOneHotEncoded.flat[indexOffset+y_data] = 1 
    return yDataOneHotEncoded

def create_batch(data,labels,batch_size,startindx):
    #batchMask = np.random.choice(train_data.shape[0],batch_size)
    if startindx + batch_size <= data.shape[0]:
        batch_x = data[startindx:startindx+batch_size].reshape(-1,data.shape[1]*data.shape[2])
        batch_y = labels[startindx:startindx+batch_size]
        batch_y = one_hot_encode_labels(batch_y)
    else:
        batch_x = data[startindx:].reshape(-1,data.shape[1]*data.shape[2])
        batch_y = labels[startindx:]
        batch_y = one_hot_encode_labels(batch_y)
    return batch_x,batch_y


def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test
