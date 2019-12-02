import numpy as np
from scipy.signal import correlate
from sklearn.utils import shuffle
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

def RemoveLargeZ(data, db, Z=12):
    minZ = float(db.Redshifts[0])
    maxZ = float(db.Redshifts[-1])
    if Z < minZ or Z > maxZ:
        raise ValueError(f"Z not in range ({minZ}, {maxZ})")
    cosmo = FlatLambdaCDM(67.8, 0.3078)
    minD, maxD, D = np.array(cosmo.comoving_distance(np.array([minZ, maxZ, Z])))
    maxIndex = int((D - minD) / (maxD - minD) * data.shape[-1] + 0.5)
    return data[..., :maxIndex]

def CutInX (data, N = 2):
    dataDim = data.shape
    bounds = list(range(0, dataDim[-2] + 1, dataDim[-2] // N))
    dataCut = data[..., bounds[0]:bounds[1], :]
    for i in range(1, N):
        dataCut = np.concatenate((dataCut, data[..., bounds[i]:bounds[i+1], :]), axis=len(dataDim)-3)
    return dataCut

def TopHat(data, Nx = 1, Nz = 2):
    """
    Basically just averaging and reducing z dimension by factor N
    Assuming the last axis is z
    """
    dataDim = data.shape
    kernelDim = (1,) * (len(dataDim) - 2) + (Nx, Nz)
    kernel = np.ones(kernelDim) / (Nx * Nz)
    #[..., ::N] -> google "Ellipsis", ... skips all dimensions in between, and ::N takes every Nth element
    return correlate(data, kernel, mode="valid", method="direct")[..., ::Nx, ::Nz]

def BoxCar3D (data, Nx = 4, Ny = 4, Nz = 4):
    dataDim = data.shape
    kernelDim = (1,) * (len(dataDim) - 3) + (Nx, Ny, Nz)
    kernel = np.ones(kernelDim) / (Nx * Ny * Nz)
    #[..., ::N] -> google "Ellipsis", ... skips all dimensions in between, and ::N takes every Nth element
    return correlate(data, kernel, mode="valid", method="direct")[..., ::Nx, ::Ny, ::Nz]

def NormalizeY(Y):
    min = np.amin(Y, axis=0)
    max = np.amax(Y, axis=0)
    return (Y - min) / (max - min), min, max

def ReshapeY(Y, Xshape):
    Y = np.broadcast_to(Y, (Xshape[1],) + Y.shape)
    Y = np.swapaxes(Y, 0, 1)
    return Y

def TDT(X, Y, pTrain, pDev, pTest, seed = 1312, WalkerSteps = 0):
    """
    Create Train Dev Test sets
    all firstly seperated in parameter space, then shuffled and saved
    """
    Y = ReshapeY(Y, X.shape)
    
    assert np.abs(pTrain + pDev + pTest - 1) < 1e-5

    n = [0, 0, 0]
    n[0] = int(X.shape[0] * pTrain)
    n[1] = int(X.shape[0] * pDev)
    n[2] = X.shape[0] - n[0] - n[1]
    indexArray = np.hstack((np.zeros(n[0], dtype=int), np.ones(n[1], dtype=int), 2*np.ones(n[2], dtype=int)))
    print(indexArray)
    RState = np.random.RandomState(seed=seed)
    indexArray = RState.permutation(indexArray)
    print(indexArray)

    if WalkerSteps:
        tdt = []
        WalkerIndex = np.zeros((X.shape[0], X.shape[1], 2))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                WalkerIndex[i, j] = [i, j]
        # print(WalkerIndex.shape)
        # WalkerIndex = ReshapeY(WalkerIndex, X.shape)
        # print(WalkerIndex.shape)
        for i in range(3):
            dX = X[indexArray==i]
            dY = Y[indexArray==i]
            dWI = WalkerIndex[indexArray==i]
            dX = dX.reshape(-1, dX.shape[-2], dX.shape[-1])
            dY = dY.reshape(-1, dY.shape[-1])
            dWI = dWI.reshape(-1, dWI.shape[-1])
            dX, dY, dWI = shuffle(dX, dY, dWI, random_state = RState)
            dX = dX.astype(np.float32)
            dY = dY.astype(np.float32)
            dWI = dWI.astype(int)
            tdt.append(dX)
            tdt.append(dY)
            tdt.append(dWI)
    else:
        tdt = []
        for i in range(3):
            dX = X[indexArray==i]
            dY = Y[indexArray==i]
            dX = dX.reshape(-1, dX.shape[-2], dX.shape[-1])
            dY = dY.reshape(-1, dY.shape[-1])
            dX, dY = shuffle(dX, dY, random_state = RState)
            dX = dX.astype(np.float32)
            dY = dY.astype(np.float32)
            tdt.append(dX)
            tdt.append(dY)
    return tdt

def basicTVT(X, Y, pTrain, pVal, pTest, seed = 1312):
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