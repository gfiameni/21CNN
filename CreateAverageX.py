import numpy as np
import sys

from database import DatabaseUtils

spa = 5 #SlicesPerAxis

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

averages = []
for i in range(database.WalkerSteps):
    Box = database.CombineBoxes(i)
    averages.append(np.nanmean(Box, axis=(0, 1), dtype='float32', keepdims=True))
    if i%100 == 0:
        print(i)
averages = np.array(averages, dtype=database.Dtype)
print(averages.shape)
np.save(f"../data/database{spa}_averages_{database.Dtype}", averages)