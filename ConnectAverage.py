import numpy as np

averages = np.load(f"../data/database5_averages_float32_00000_01000.npy")
for i in range(1, 10):
    avg = np.load(f"../data/database5_averages_float32_{i*1000:05d}_{(i+1)*1000:05d}.npy")
    averages = np.concatenate((averages, avg), axis=0)
averages = averages.astype('float32')
np.save(f"../data/database5_averages_float32", averages)