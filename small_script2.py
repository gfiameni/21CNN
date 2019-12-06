import numpy as np
import sys
import os
import json

from database import DatabaseUtils
from CNN.formatting import Filters
import json

#define path to database, send to program as parameters if different from default
BoxesPath = "/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/LightConeBoxes"
ParametersPath = "/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/GridPositions"
Redshifts = ['006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42503', \
            '013.25424', '015.51874', '018.36856', '022.02434', '026.82138', '033.28927', '034.50984']
Parameters = ["ZETA", "TVIR_MIN", "L_X", "NU_X_THRESH"]
database = DatabaseUtils.Database(Parameters, Redshifts, BoxesPath, ParametersPath)

Zmax = 30

avg = np.load("../data/averages_float32.npy")
print(avg.shape)
avg = Filters.RemoveLargeZ(avg, database, Z=Zmax)
print(avg.shape)
avg = Filters.BoxCar3D(avg, Nx = 1, Ny = 1, Nz = 4)
print(avg.shape)

for i in range(10000):
    box = np.load(f"../data/database3D/{i}_boxcar444_float32.npy")
    box *= 300
    box += -250
    box -= avg[i]
    box = box.astype(np.float32)
    np.save(f"../data/database3D/{i}_boxcar444_meanRemoved_float32.npy", box)
    if i%100 == 0:
        print(i)

