import numpy as np

data = np.load("data3D_boxcar444_float32.npy")
total = []

total.append(data[:, :25, :25, :])
total.append(data[:, 25:, :25, :])
total.append(data[:, :25, 25:, :])
total.append(data[:, 25:, 25:, :])

total = np.array(total, dtype=np.float32)
np.save("data3D_boxcar444_sliced22_float32.npy", total)