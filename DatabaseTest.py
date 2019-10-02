import numpy as np
# from database import DatabaseUtils
import matplotlib.pyplot as plt
import matplotlib
import json
import sys

from database import DatabaseUtils
from CNN.formatting import Filters

EoR_colour = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', [(0, 'white'),(0.33, 'yellow'),(0.5, 'orange'),(0.68, 'red'),(0.833, 'black'),(0.87, 'blue'),(1, 'cyan')])
plt.register_cmap(cmap=EoR_colour)

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

DataFilepath = "../data/train/"
DataXname = "X_0.8_tophat22_Z12_database5_float32.npy"
DataYname = "Y_0.8_tophat22_Z12_database5_float32.npy"
Yparamsname = "../data/databaseParams_min_max.txt"
WalkerIndexname = "X_0.8_WalkerIndex.npy"
# DataYname = f"databaseParams_{database.Dtype}.npy"

DataX = np.load(DataFilepath+DataXname)
DataY = np.load(DataFilepath+DataYname)
WalkerIndex = np.load(DataFilepath+WalkerIndexname)
with open(Yparamsname, "r") as f:
    Yparams = json.loads(f.read())
Yparams['min'] = np.array(Yparams['min'])
Yparams['max'] = np.array(Yparams['max'])

fig=plt.figure(figsize=(10, 10))
images = DataX.shape[0]

print(Yparams['parameters'])
print(Yparams['min'])
print(Yparams['max'], '\n')
for i in range(10):
    fig.add_subplot(10, 1, i+1)
    plt.pcolormesh(DataX[images*i//10], vmin = 0, vmax = 1, cmap=EoR_colour,shading='gouraud')
    # plt.imshow(DataX[images*i//10], cmap="gray")
    print(DataY[images*i//10] * (Yparams['max'] - Yparams['min']) + Yparams['min'], WalkerIndex[images*i//10])
plt.savefig('Database_train_tophat_withmean.pdf')
plt.close()

fig=plt.figure(figsize=(10, 20))
for i in range(10):
    fig.add_subplot(10, 1, i+1)
    WI = WalkerIndex[images*i//10]
    Box = database.CombineBoxes(WI[0], 5)
    if(WI[1] < 5):
        slice = Box[WI[1] * Box.shape[0] // 5, :, :]
    else:
        slice = Box[:, (WI[1]-5) * Box.shape[1] // 5, :]
    plt.pcolormesh(slice, vmin = -250, vmax = 50, cmap=EoR_colour,shading='gouraud')
    # plt.imshow(DataX[images*i//10], cmap="gray")
plt.savefig('Real_Database_train.pdf')
plt.close()

