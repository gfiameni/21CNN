import numpy as np
# from database import DatabaseUtils
import matplotlib.pyplot as plt
import matplotlib

EoR_colour = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', [(0, 'white'),(0.33, 'yellow'),(0.5, 'orange'),(0.68, 'red'),(0.833, 'black'),(0.87, 'blue'),(1, 'cyan')])
plt.register_cmap(cmap=EoR_colour)


# Redshifts = ['006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42503', \
#             '013.25424', '015.51874', '018.36856', '022.02434', '026.82138', '033.28927', '034.50984']
# Parameters = ["ZETA", "TVIR_MIN", "L_X", "NU_X_THRESH"]
# database = DatabaseUtils.Database(Parameters, Redshifts)

# spa = 5

DataFilepath = "../data/test/"
DataXname = "X_0.1_tophat22_Z12_database5_float32.npy"
DataYname = "Y_0.1_tophat22_Z12_database5_float32.npy"
# DataYname = f"databaseParams_{database.Dtype}.npy"

DataX = np.load(DataFilepath+DataXname)
DataY = np.load(DataFilepath+DataYname)

fig=plt.figure(figsize=(10, 10))
images = DataX.shape[0]
for i in range(10):
    fig.add_subplot(10, 1, i+1)
    plt.pcolormesh(DataX[images*i//10], vmin = 0, vmax = 1, cmap=EoR_colour,shading='gouraud')
    # plt.imshow(DataX[images*i//10], cmap="gray")
    print(DataY[images*i//10])
plt.savefig('Database_tophat_withmean.pdf')