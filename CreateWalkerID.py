import numpy as np
from database import DatabaseUtils
from CNN.formatting import Filters

DataFilepath = "../data/"
zeroX = np.zeros((10000, 20, 1, 1))
zeroY = np.zeros((10000, 1))

pTrain = 0.8
pDev = 0.1
pTest = 0.1
_, _, trainWI, _, _, devWI, _, _, testWI = Filters.TDT(zeroX, zeroY, pTrain, pDev, pTest, WalkerSteps=10000)
print("train and test created, now saving...")

np.save(f"{DataFilepath}train/X_{pTrain:.1f}_WalkerIndex", trainWI)
np.save(f"{DataFilepath}test/X_{pTest:.1f}_WalkerIndex", testWI)
np.save(f"{DataFilepath}dev/X_{pTest:.1f}_WalkerIndex", devWI)