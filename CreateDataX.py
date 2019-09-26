import matplotlib.pyplot as plt
import numpy as np
import sys

from database import DatabaseUtils

spa = 10

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

##############TESTING##############
# Box = database.CombineBoxes(9999, 12)
# BoxSlices = DatabaseUtils.SliceBoxNTimesXY(Box, 5)
# fig=plt.figure(figsize=(10, 10))
# for i in range(1, 10 + 1):
#     fig.add_subplot(10, 1, i)
#     plt.imshow(BoxSlices[i-1], cmap="gray")
# plt.savefig('/home/dprelogovic/Documents/proba_slices.pdf')

TotalData = DatabaseUtils.CreateSlicedData(database, SlicesPerAxis = spa)
print(TotalData.shape)
np.save(f"../data/database{spa}", TotalData)