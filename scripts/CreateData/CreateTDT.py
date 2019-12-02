#########################################################
########## Manipulating original sliced data and creating
########## Train Test Dev data
#########################################################
import numpy as np
from database import DatabaseUtils
from CNN.formatting import Filters
import json

RemoveMean = False
if RemoveMean:
    rm = "_meanZ"
else:
    rm = "" #used only in file naming

#define database, paths not important at the moment
Redshifts = ['006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42503', \
            '013.25424', '015.51874', '018.36856', '022.02434', '026.82138', '033.28927', '034.50984']
Parameters = ["ZETA", "TVIR_MIN", "L_X", "NU_X_THRESH"]
database = DatabaseUtils.Database(Parameters, Redshifts)

spa = 5

DataFilepath = "../data/"
DataXname = f"database{spa}_{database.Dtype}.npy"
DataYname = f"databaseParams_{database.Dtype}.npy"
AverageXname = f"database{spa}_averages_{database.Dtype}_clipped.npy"

deltaTmin = -250
deltaTmax = 50
tophat = [2, 2]
Zmax = 30

#loading X and Y and averages
DataY = np.load(DataFilepath+DataYname)
DataX = np.load(DataFilepath+DataXname)
AverageX = np.load(DataFilepath+AverageXname)
print(f"data loaded X={DataX.shape}, Y={DataY.shape}, avgX={AverageX.shape}")

#normalizing Y -> every parameter in [0, 1]
Ybackup = {}
Ybackup['parameters'] = database.Parameters
DataY, Ybackup['min'], Ybackup['max'] = Filters.NormalizeY(DataY)
Ybackup['min'] = Ybackup['min'].tolist()
Ybackup['max'] = Ybackup['max'].tolist()
with open(DataFilepath+"databaseParams_min_max.txt", 'w') as f:
    json.dump(Ybackup, f)

#cutting X and AverageX for large Z
DataX = Filters.RemoveLargeZ(DataX, database, Z=Zmax)
AverageX = Filters.RemoveLargeZ(AverageX, database, Z=Zmax)
print(f"Remove large Z {DataX.shape}")
DataX = Filters.CutInX(DataX, N=2)
print(f"Cut x-dim in half {DataX.shape}")

#filtering X
np.nan_to_num(DataX, copy=False, nan=deltaTmin, posinf=deltaTmax, neginf=deltaTmin)
print(f"NaN's and infinities set to {deltaTmin}, {deltaTmax}")
np.clip(DataX, deltaTmin, deltaTmax, out=DataX)
print("large values clipped")

#removing mean for every Z for all images
if RemoveMean:
    DataX = DataX - AverageX
    print("mean removed")
DataX = Filters.TopHat(DataX, Nx = tophat[0], Nz = tophat[1])
print(f"Top Hat 2, 2 {DataX.shape}")
if RemoveMean:
    deltaTmin = np.amin(DataX)
    deltaTmax = np.amax(DataX)
DataX = (DataX - deltaTmin) / (deltaTmax - deltaTmin)
print("X normalized")

np.save(f"{DataFilepath}database{spa}_{database.Dtype}_tophat{tophat[0]}{tophat[1]}_Z{Zmax}{rm}.npy", DataX)
np.save(f"{DataFilepath}databaseParams_{database.Dtype}_tophat{tophat[0]}{tophat[1]}_Z{Zmax}{rm}.npy", DataY)
print("processed X and Y saved before dividing into train dev test")

pTrain = 0.8
pDev = 0.1
pTest = 0.1
trainX, trainY, devX, devY, testX, testY = Filters.TDT(DataX, DataY, pTrain, pDev, pTest)
print("train and test created, now saving...")

np.save(f"{DataFilepath}train/X_{pTrain:.1f}_tophat{tophat[0]}{tophat[1]}_Z{Zmax}{rm}_{DataXname}", trainX)
np.save(f"{DataFilepath}train/Y_{pTrain:.1f}_tophat{tophat[0]}{tophat[1]}_Z{Zmax}{rm}_{DataXname}", trainY)
np.save(f"{DataFilepath}test/X_{pTest:.1f}_tophat{tophat[0]}{tophat[1]}_Z{Zmax}{rm}_{DataXname}", testX)
np.save(f"{DataFilepath}test/Y_{pTest:.1f}_tophat{tophat[0]}{tophat[1]}_Z{Zmax}{rm}_{DataXname}", testY)
np.save(f"{DataFilepath}dev/X_{pTest:.1f}_tophat{tophat[0]}{tophat[1]}_Z{Zmax}{rm}_{DataXname}", devX)
np.save(f"{DataFilepath}dev/Y_{pTest:.1f}_tophat{tophat[0]}{tophat[1]}_Z{Zmax}{rm}_{DataXname}", devY)
