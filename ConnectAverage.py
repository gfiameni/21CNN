import numpy as np

averages = []
for i in range(10000):
    avg = np.load(f"../data/database3D/{i}_boxcar444_meanRemoved_float32.npy")
    a = []
    a.append(avg[:25, :25, :])
    a.append(avg[25:, :25, :])
    a.append(avg[:25, 25:, :])
    a.append(avg[25:, 25:, :])
    a = np.array(a, dtype='float32')
    averages.append(a)
    if i%100 == 0:
        print(i)
averages = np.array(averages, dtype = 'float32')
np.save(f"../data/data3D_boxcar444_sliced22_meanRemoved_float32.npy", averages)
