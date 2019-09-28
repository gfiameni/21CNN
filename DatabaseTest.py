import numpy as np
from database import DatabaseUtils
import matplotlib.pyplot as plt

Redshifts = ['006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42503', \
            '013.25424', '015.51874', '018.36856', '022.02434', '026.82138', '033.28927', '034.50984']
Parameters = ["ZETA", "TVIR_MIN", "L_X", "NU_X_THRESH"]
database = DatabaseUtils.Database(Parameters, Redshifts)

spa = 5

DataFilepath = "../data/"
DataXname = f"database{spa}_{database.Dtype}.npy"
# DataYname = f"databaseParams_{database.Dtype}.npy"

DataX = np.load(DataFilepath+DataXname)


fig=plt.figure(figsize=(10, 10))
for i in range(1, 10 + 1):
    fig.add_subplot(10, 1, i)
    plt.imshow(DataX[100*i, 2], cmap="gray")
plt.savefig('DatabaseTest.pdf')