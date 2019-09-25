import numpy as np
from scipy.signal import correlate

def RemoveLargeZ(data, db, Z = 12):
    minZ = float(db.Redshifts[0])
    maxZ = float(db.Redshifts[-1])
    if Z < minZ or Z > maxZ:
        raise ValueError("Z not in range")
    maxIndex = int((Z - minZ) / (maxZ - minZ) * data.shape[-1] + 0.5)
    return data[..., :maxIndex]

def CutInX (data, N = 2):
    dataDim = data.shape
    bounds = list(range(0, dataDim[2] + 1, dataDim[2] // N))
    dataCut = data[:, :, bounds[0]:bounds[1], :]
    for i in range(1, N):
        dataCut = np.concatenate((dataCut, data[:, :, bounds[i], bounds[i+1], :]), axis=1)
    return dataCut

def TopHat(data, N = 2):
    """
    Basically just averaging and reducing z dimension by factor N
    Assuming the last axis is z
    """
    dataDim = data.shape
    kernelDim = (1,) * (len(dataDim) - 1) + (N,)
    kernel = np.ones(kernelDim) / N
    #[..., ::N] -> google "Ellipsis", ... skips all dimensions but the last one, and ::N takes every Nth 
    return correlate(data, kernel, mode="valid", method="direct")[..., ::N]

