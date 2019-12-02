import numpy as np
from CNN.formatting import Filters
import json
DataLoc = "/astro/home/david.prelogovic/data/"
XName = "data3D_boxcar444_sliced22_float32.npy"
YName = "databaseParams_float32.npy"
YbackupName = "databaseParams_min_max.txt"


Y = np.load(DataLoc+YName)
print(Y.shape)
with open(DataLoc+YbackupName, "r") as f:
    Params = json.load(f)
Params["min"] = np.array(Params["min"], dtype = np.float32)
Params["max"] = np.array(Params["max"], dtype = np.float32)
Y = (Y - Params["min"]) / (Params["max"] - Params["min"])
shape = Y.shape
Y = Y[..., np.newaxis]
Y = np.broadcast_to(Y, shape + (4,))
Y = np.swapaxes(Y, -1, -2)

X = np.load(DataLoc+XName)
X = np.swapaxes(X, 2, 4)
X = X.reshape(-1, X.shape[-3], X.shape[-2], X.shape[-1])
Y = Y.reshape(-1, Y.shape[-1])
print("X.shape", X.shape)
print("Y.shape", Y.shape)

XX = {}
YY = {}

XX["train"], YY["train"], XX["val"], YY["val"], XX["test"], YY["test"] = Filters.basicTVT(X, Y, 0.8, 0.1, 0.1)

for sort in ['train', 'val', 'test']:
    XX[sort] = XX[sort].astype(np.float32)
    YY[sort] = YY[sort].astype(np.float32)
    np.save(f"{DataLoc}{sort}/X_{XName}", XX[sort])
    np.save(f"{DataLoc}{sort}/Y_{XName}", YY[sort])
