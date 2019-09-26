import numpy as np
from scipy.signal import correlate

def RemoveLargeZ(data, db, Z=12):
    #this should be fixed!, as data is not linear in Z
    minZ = float(db.Redshifts[0])
    maxZ = float(db.Redshifts[-1])
    if Z < minZ or Z > maxZ:
        raise ValueError(f"Z not in range ({minZ}, {maxZ})")
    maxIndex = int((Z - minZ) / (maxZ - minZ) * data.shape[-1] + 0.5)
    return data[..., :maxIndex]

def CutInX (data, N = 2):
    dataDim = data.shape
    bounds = list(range(0, dataDim[2] + 1, dataDim[2] // N))
    dataCut = data[:, :, bounds[0]:bounds[1], :]
    for i in range(1, N):
        dataCut = np.concatenate((dataCut, data[:, :, bounds[i]:bounds[i+1], :]), axis=1)
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

def NormalizeY(Y):
    min = np.amin(Y, axis=0)
    max = np.amax(Y, axis=0)
    return (Y - min) / (max - min)

def ReshapeY(Y, Xshape):
    Y = np.broadcast_to(Y, (Xshape[1],) + Y.shape)
    Y = np.swapaxes(Y, 0, 1)
    return Y

def TDT(X, Y, pTrain, pDev, pTest, seed = 1312):
    assert np.abs(pTrain + pDev + pTest - 1) < 1e-3
    assert X.shape[0] == Y.shape[0]

    n = [0, 0, 0]
    n[0] = int(X.shape[0] * pTrain)
    n[1] = int(X.shape[0] * pDev)
    n[2] = X.shape[0] - n[0] - n[1]
    indexArray = np.hstack((np.zeros(n[0], dtype=int), np.ones(n[1], dtype=int), 2*np.ones(n[2], dtype=int)))

    RState = np.random.RandomState(seed=seed)
    indexArray = RState.permutation(indexArray)

    tdt = []
    for i in range(3):
        tdt.append(X[indexArray==i])
        tdt.append(Y[indexArray==i])
    
    return tdt
