import numpy as np

data = np.load("data3D_boxcar444_float32.npy")
print(data.shape)
total = []

total.append(data[:, :25, :25, :])
total.append(data[:, 25:, :25, :])
total.append(data[:, :25, 25:, :])
total.append(data[:, 25:, 25:, :])

total = np.array(total, dtype=np.float32)
print(total.shape)
total = np.swapaxes(total, 0, 1)
print(total.shape)
np.save("data3D_boxcar444_sliced22_float32.npy", total)