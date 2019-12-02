import numpy as np
import sys

from database import DatabaseUtils

i = int(sys.argv[1])
deltaTmin = -250
deltaTmax = 50

#define path to database, send to program as parameters if different from default

BoxesPath = "/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/LightConeBoxes"
ParametersPath = "/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/GridPositions"

Redshifts = ['006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42503', \
            '013.25424', '015.51874', '018.36856', '022.02434', '026.82138', '033.28927', '034.50984']
Parameters = ["ZETA", "TVIR_MIN", "L_X", "NU_X_THRESH"]

database = DatabaseUtils.Database(Parameters, Redshifts, BoxesPath, ParametersPath)

Box = database.CombineBoxes(i)
np.nan_to_num(Box, copy=False, nan=deltaTmin, posinf=deltaTmax, neginf=deltaTmin)
np.clip(Box, deltaTmin, deltaTmax, out=Box)
averages = np.mean(Box, axis=(0, 1), dtype=database.Dtype, keepdims=True)
np.save(f"../data/databaseMean/averages_{i}_{database.Dtype}", averages)
