import numpy as np
from sklearn.utils import shuffle
from src.py21cnn import utilities

def basicTVT(X, Y, pTrain, pVal, pTest, seed = 1312):
    assert np.abs(pTrain + pVal + pTest - 1) < 1e-5
    n = [0, 0, 0]
    n[0] = int(X.shape[0] * pTrain)
    n[1] = int(X.shape[0] * pVal)
    n[2] = X.shape[0] - n[0] - n[1]
    indexArray = np.hstack((np.zeros(n[0], dtype=int), np.ones(n[1], dtype=int), 2*np.ones(n[2], dtype=int)))
    
    if isinstance(seed, int):
        RState = np.random.RandomState(seed=seed)
    elif isinstance(seed, np.random.RandomState):
        RState = seed
    else:
        raise TypeError('seed should be int or numpy.random.RandomState instance')
    
    indexArray = RState.permutation(indexArray)
    
    dX = {}
    dY = {}
    for key, i in zip(['train', 'val', 'test'], [0, 1, 2]):
        dX[key] = X[indexArray==i]
        dY[key] = Y[indexArray==i]
        dX[key], dY[key] = shuffle(dX[key], dY[key], random_state = RState)
        dX[key] = dX[key].astype(np.float32)
        dY[key] = dY[key].astype(np.float32)
        # tdt.append(dX)
        # tdt.append(dY)
    return dX, dY

DataFilepath = "data/"
DataXname = f"data3D_boxcar444_sliced22_meanRemoved_float32.npy"
DataYname = f"databaseParams_float32.npy"
dataY = np.load(DataFilepath+DataYname)
dataX = np.load(DataFilepath+DataXname)

shapeY = dataY.shape
dataY = dataY[..., np.newaxis]
dataY = np.broadcast_to(dataY, shapeY + (4,))
dataY = np.swapaxes(dataY, -1, -2)

RState = np.random.RandomState(seed=1312)
X, Y = basicTVT(dataX, dataY, 0.8, 0.1, 0.1, RState)
for key in X.keys():
    shapeX = X[key].shape
    shapeY = Y[key].shape
    X[key] = np.reshape(X[key], (shapeX[0] * shapeX[1],) + shapeX[2:])
    Y[key] = np.reshape(Y[key], (shapeY[0] * shapeY[1],) + shapeY[2:])
    X[key], Y[key] = shuffle(X[key], Y[key], random_state = RState)

    if key == 'train':
        msX = {'mean': np.mean(X[key]), 'std': np.std(X[key])}
        msY = {'mean': np.mean(Y[key], axis=0), 'std': np.std(Y[key], axis=0)}
print(msX, msY)
for key in X.keys():
    X[key] = (X[key] - msX['mean']) / msX['std']
    Y[key] = (Y[key] - msY['mean']) / msY['std']
    print(f'X[{key}] mean std: {np.mean(X[key])}, {np.std(X[key])}')
    print(X[key].shape)
    print(f'Y[{key}] mean std: {np.mean(Y[key])}, {np.std(Y[key])}')
    print(Y[key].shape)


Data = utilities.Data(filepath=DataFilepath, dimensionality=3, X=X, Y=Y)
Data.formatting.append('TVT_parameterwise')

Data.saveTVT()