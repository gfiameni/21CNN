#########################################################
########## Manipulating original sliced data and creating
########## Train Test Dev data
#########################################################
import numpy as np
from database import DatabaseUtils
from CNN.formatting import Filters
import json

#define database, paths not important at the moment
Redshifts = ['006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42503', \
            '013.25424', '015.51874', '018.36856', '022.02434', '026.82138', '033.28927', '034.50984']
Parameters = ["ZETA", "TVIR_MIN", "L_X", "NU_X_THRESH"]
database = DatabaseUtils.Database(Parameters, Redshifts)

spa = 5

DataFilepath = "../data/"
DataXname = f"database{spa}_{database.Dtype}.npy"
DataYname = f"databaseParams_{database.Dtype}.npy"
AverageXname = f"database{spa}_averages_{database.Dtype}.npy"

deltaTmin = -250
deltaTmax = 50

#loading X and Y and averages
DataY = np.load(DataFilepath+DataYname)
DataX = np.load(DataFilepath+DataXname)
AverageX = np.load(DataFilepath+AverageXname)
print(f"data loaded X={DataX.shape}, Y={DataY.shape}, avgX={AverageX.shape}")

#normalizing Y -> every parameter in [0, 1]
Ybackup = {}
Ybackup['parameters'] = database.Parameters
DataY, Ybackup['min'], Ybackup['max'] = Filters.NormalizeY(DataY)
with open(DataFilepath+"databaseParams_min_max.txt", 'w') as f:
    json.dump(Ybackup, f)

#cutting X and AverageX
DataX = Filters.RemoveLargeZ(DataX, database, Z=12)
AverageX = Filters.RemoveLargeZ(AverageX, database, Z=12)
print(f"Remove large Z {DataX.shape}")
DataX = Filters.CutInX(DataX, N=2)
print(f"Cut x-dim in half {DataX.shape}")

#filtering X
np.nan_to_num(DataX, copy=False, nan=deltaTmin, posinf=deltaTmax, neginf=deltaTmin)
print(f"NaN's and infinities set to {deltaTmin}, {deltaTmax}")
np.clip(DataX, deltaTmin, deltaTmax, out=DataX)
print("large values clipped")
DataX = Filters.TopHat(DataX, Nx = 2, Nz = 2)
print(f"Top Hat 2, 2 {DataX.shape}")
#removing mean for every Z for all images
DataX = DataX - AverageX[:,np.newaxis] #not sure if I need to add axis or will it be broadcasted by itself
print("mean removed")
#normalizing X
deltaTmin = np.amin(DataX)
deltaTmax = np.amax(DataX)
DataX = (DataX - deltaTmin) / (deltaTmax - deltaTmin)
print("X normalized")

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