import numpy as np
# from scipy.signal import correlate
from sklearn.utils import shuffle

def basicTDT(X, Y, pTrain, pVal, pTest, seed = 1312):
    assert np.abs(pTrain + pVal + pTest - 1) < 1e-5
    n = [0, 0, 0]
    n[0] = int(X.shape[0] * pTrain)
    n[1] = int(X.shape[0] * pVal)
    n[2] = X.shape[0] - n[0] - n[1]
    indexArray = np.hstack((np.zeros(n[0], dtype=int), np.ones(n[1], dtype=int), 2*np.ones(n[2], dtype=int)))
    RState = np.random.RandomState(seed=seed)
    indexArray = RState.permutation(indexArray)
    
    tdt = []
    for i in range(3):
        dX = X[indexArray==i]
        dY = Y[indexArray==i]
        dX, dY = shuffle(dX, dY, random_state = RState)
        dX = dX.astype(np.float32)
        dY = dY.astype(np.float32)
        tdt.append(dX)
        tdt.append(dY)
    return tdt