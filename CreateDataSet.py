import matplotlib.pyplot as plt
import numpy as np
import sys

import DatabaseUtils


#define path to database, send to program as parameters if different from default
if len(sys.argv) == 1:
    BoxesPath = "/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/LightConeBoxes"
else:
    BoxesPath = sys.argv[1]
if len(sys.argv) <= 2:
    ParametersPath = "/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/GridPositions"
else:
    ParametersPath = sys.argv[2]

Redshifts = ['006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42503',
            '013.25424', '015.51874', '018.36856', '022.02434', '026.82138', '033.28927', '034.50984']

database = DatabaseUtils.Database(BoxesPath, ParametersPath, Redshifts)

##############TESTING##############
# Box = database.CombineBoxes(9999, 12)
# BoxSlices = DatabaseUtils.SliceBoxNTimesXY(Box, 5)
# fig=plt.figure(figsize=(10, 10))
# for i in range(1, 10 + 1):
#     fig.add_subplot(10, 1, i)
#     plt.imshow(BoxSlices[i-1], cmap="gray")
# plt.savefig('/home/dprelogovic/Documents/proba_slices.pdf')

spa = 5
def CreateSlicedData(db, SlicesPerAxis = spa):
    """
    Creating general sliced cubes without post or preprocessing
    """
    Box = db.CombineBoxes(0)
    # Box = db.CombineBoxes(9999)
    BoxSlices = DatabaseUtils.SliceBoxNTimesXY(Box, SlicesPerAxis)
    # DataShape = (10,) + BoxSlices.shape
    DataShape = (db.WalkerSteps,) + BoxSlices.shape #adding one WalkerSteps to dimension of FinalData
    FinalData = np.empty(DataShape, dtype = BoxSlices.dtype)
    FinalData[0] = BoxSlices

    for i in range(1, db.WalkerSteps):
    # for i in range(1, 10):
        # Box = db.CombineBoxes(9999)
        Box = db.CombineBoxes(i)
        BoxSlices = DatabaseUtils.SliceBoxNTimesXY(Box, SlicesPerAxis)
        FinalData[i] = BoxSlices
        if i%100 == 0:
            print(i)

    return FinalData

TotalData = CreateSlicedData(database)
print(TotalData.shape)
np.save(f"database{spa}", TotalData)