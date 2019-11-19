import numpy as np
import sys

from database import DatabaseUtils
from CNN.formatting import Filters
import json

#define path to database, send to program as parameters if different from default
if len(sys.argv) == 1:
    BoxesPath = "/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/LightConeBoxes"
else:
    BoxesPath = sys.argv[1]
if len(sys.argv) <= 2:
    ParametersPath = "/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/GridPositions"
else:
    ParametersPath = sys.argv[2]
Redshifts = ['006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42503', \
            '013.25424', '015.51874', '018.36856', '022.02434', '026.82138', '033.28927', '034.50984']
Parameters = ["ZETA", "TVIR_MIN", "L_X", "NU_X_THRESH"]

database = DatabaseUtils.Database(Parameters, Redshifts, BoxesPath, ParametersPath)
deltaTmin = -250
deltaTmax = 50
BoxCar = [4, 4, 4]
Zmax = 30


FinalData = []
for i in range(database.WalkerSteps):
    Box = database.CombineBoxes(i)
    Box = Filters.RemoveLargeZ(Box, database, Z=Zmax)
    np.nan_to_num(Box, copy=False, nan=deltaTmin, posinf=deltaTmax, neginf=deltaTmin)
    np.clip(Box, deltaTmin, deltaTmax, out=Box)
    Box = Filters.BoxCar3D(Box, Nx = BoxCar[0], Ny = BoxCar[1], Nz = BoxCar[2])
    Box = (Box - deltaTmin) / (deltaTmax - deltaTmin)
    FinalData.append(np.array(Box, dtype=database.Dtype))
    if i%100 == 0:
        print(i)
FinalData = np.array(FinalData, dtype=database.Dtype)

print(FinalData.shape)
np.save(f"../data/database3D_boxcar{BoxCar[0]}{BoxCar[1]}{BoxCar[2]}_{database.Dtype}", FinalData)