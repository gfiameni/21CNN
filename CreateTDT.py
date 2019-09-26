#########################################################
########## Manipulating original sliced data and creating
########## Train Test Dev data
#########################################################
import numpy as np
from database import DatabaseUtils
from CNN.formatting import Filters

DataFilepath = "../data/"
DataXname = "database5.npy"
DataYname = "databaseParams.npy"

deltaTmin = -250
deltaTmax = 50

#define database, paths not important at the moment
Redshifts = ['006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42503', \
            '013.25424', '015.51874', '018.36856', '022.02434', '026.82138', '033.28927', '034.50984']
Parameters = ["ZETA", "TVIR_MIN", "L_X", "NU_X_THRESH"]
database = DatabaseUtils.Database(Parameters, Redshifts)

#loading X and Y
DataY = np.load(DataFilepath+DataYname)
DataX = np.load(DataFilepath+DataXname)
print(f"data loaded X={DataX.shape}, Y={DataY.shape}")

#normalizing Y -> every parameter in [0, 1]
DataY = Filters.NormalizeY(DataY)

#cutting X
DataX = DataX[..., :1000] #should be replaced in future with Filters.RemoveLargeZ
print(f"Remove large Z {DataX.shape}")
DataX = Filters.CutInX(DataX, N=2)
print(f"Cut x-dim in half {DataX.shape}")

#reshaping X and Y, so they have shape of (N, Nx, Nz)
DataY = Filters.ReshapeY(DataY, DataX.shape)
DataX = DataX.reshape(-1, DataX.shape[-2], DataX.shape[-1])
DataY = DataY.reshape(-1, DataY.shape[-1])

#filtering X
np.nan_to_num(DataX, copy=False)
print("NaN's set to 0")
DataX = Filters.TopHat(DataX, Nx = 2, Nz = 2)
print(f"Top Hat 2, 2 {DataX.shape}")
np.clip(DataX, deltaTmin, deltaTmax, out=DataX)
print("large values clipped")
#removing mean for every Z for all images
DataX = DataX - np.mean(DataX, axis=1, keepdims=True) #need to keepdims so that broadcasting works
#normalizing X
deltaTmin = np.amin(DataX)
deltaTmax = np.amax(DataX)
DataX = (DataX - deltaTmin) / (deltaTmax - deltaTmin)

pTrain = 0.8
pDev = 0.1
pTest = 0.1
trainX, trainY, devX, devY, testX, testY = Filters.TDT(DataX, DataY, pTrain, pDev, pTest)
print("train and test created, now saving...")

np.save(f"{DataFilepath}train/X_tophat22_Z12_meanZ_{DataXname}", trainX)
np.save(f"{DataFilepath}train/Y_tophat22_Z12_meanZ_{DataXname}", trainY)
np.save(f"{DataFilepath}test/X_tophat22_Z12_meanZ_{DataXname}", testX)
np.save(f"{DataFilepath}test/Y_tophat22_Z12_meanZ_{DataXname}", testY)
np.save(f"{DataFilepath}dev/X_tophat22_Z12_meanZ_{DataXname}", devX)
np.save(f"{DataFilepath}dev/Y_tophat22_Z12_meanZ_{DataXname}", devY)