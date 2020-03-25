import argparse
parser = argparse.ArgumentParser(prog = 'Run Model')
parser.add_argument('--WalkerID', type=int, choices=range(10000), default=0)
parser.add_argument('--uv_filepath', type=str, default='uv.npy')
parser.add_argument('--saving_location', type=str, default='')
parser.add_argument('--averages_fstring', type=str, default='averages_{}_float32.npy')
parser.add_argument('--BoxesPath', type=str, default="/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/LightConeBoxes")
parser.add_argument('--ParametersPath', type=str, default="/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/GridPositions")
parser.add_argument('--depth_mhz', type=int, default = 0) # if depth==0 calculate it from cube, else fix it to given value
# parser.add_argument('--uv_treshold', type=int, default = 1) # taking in account only baselines which were visited uv_treshold amount of times 
parser.add_argument('--SKA_observation_time', type=int, default = 1000)
inputs = parser.parse_args()

import numpy as np
import sys
import os
from src.py21cnn.database import DatabaseUtils
from src.py21cnn.formatting import Filters
import json

#define path to database, send to program as parameters if different from default
Redshifts = ['006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42503', \
            '013.25424', '015.51874', '018.36856', '022.02434', '026.82138', '033.28927', '034.50984']
Parameters = ["ZETA", "TVIR_MIN", "L_X", "NU_X_THRESH"]
database = DatabaseUtils.Database(Parameters, Redshifts, inputs.BoxesPath, inputs.ParametersPath)
deltaTmin = -250
deltaTmax = 50
Zmax = 30
print("loading lightcone")
Box = database.CombineBoxes(inputs.WalkerID)
print("removing large Z")
Box = Filters.RemoveLargeZ(Box, database, Z=Zmax)
print("removing NaNs")
np.nan_to_num(Box, copy=False, nan=deltaTmin, posinf=deltaTmax, neginf=deltaTmin)
print("clipping large values")
np.clip(Box, deltaTmin, deltaTmax, out=Box)

BoxAverage = np.load(inputs.averages_fstring.format(inputs.WalkerID))
print("removing large Z for average")
BoxAverage = Filters.RemoveLargeZ(BoxAverage, database, Z=Zmax)

Box -= BoxAverage

import tools21cm as t2c
t2c.const.set_hubble_h(0.678)
t2c.const.set_omega_matter(0.308)
t2c.const.set_omega_baryon(0.048425)
t2c.const.set_omega_lambda(0.692)
t2c.const.set_ns(0.968)
t2c.const.set_sigma_8(0.815)

uv = np.load(inputs.uv_filepath)
N_ant = 512 #or 513?
d0 = t2c.cosmology.z_to_cdist(float(Redshifts[0]))
cdist = np.array(range(Box.shape[-1] + 1))*1.5 + d0 #adding one more redshit to the end
redshifts = t2c.cosmology.cdist_to_z(cdist)
redshifts_mean = (redshifts[:-1] + redshifts[1:]) / 2

def noise(Box, depth_mhz, uv, seed_index):
    finalBox = []
    for i in range(Box.shape[-1]):
        if depth_mhz == 0:
            depth_mhz = t2c.cosmology.z_to_nu(redshifts[i]) - t2c.cosmology.z_to_nu(redshifts[i+1])
        noise = t2c.noise_model.noise_map(ncells=200,
                                          z=redshifts_mean[i],
                                          depth_mhz=depth_mhz,
                                          obs_time=inputs.SKA_observation_time,
                                          boxsize=300, 
                                          uv_map=uv[..., i],
                                          N_ant=N_ant,
                                          seed = 1000000*i + 100*inputs.WalkerID + seed_index, #last index is noise number index
                                          ) # I've corrected the function so it returns noise in uv, not in real space
        noise = t2c.telescope_functions.jansky_2_kelvin(noise, redshifts_mean[i])
        noise[uv[..., i]==0] = 0
        finalBox.append(noise)
    finalBox = np.moveaxis(np.array(finalBox), 0, -1)
#     print(finalBox.shape)
    return finalBox
def noise_n_signal(Box, depth_mhz, uv, seed_index):
    Noise = noise(Box, depth_mhz, uv, seed_index)
    finalBox = np.fft.fft2(Box, axes=(0, 1)) + Noise
    finalBox[uv==0] = 0
    return np.real(np.fft.ifft2(finalBox, axes=(0, 1)))
def smooth(Box, max_baseline = 2.):
    finalBox, _  = t2c.smoothing.smooth_lightcone(Box, z_array=redshifts_mean, box_size_mpc=300, max_baseline = max_baseline)
    return finalBox

for seed_indx in range(5):
    x = noise_n_signal(Box, 0, uv, seed_index = seed_indx)
    x = smooth(x)
    x = Filters.BoxCar3D(x)
    np.save(f"{inputs.saving_location}/lightcone_depthMhz_{inputs.depth_mhz}_SKAobstime_{inputs.SKA_observation_time}_walker_{inputs.WalkerID:04d}_seed_{seed_indx}_slice_0.npy", x[:25, :25])
    np.save(f"{inputs.saving_location}/lightcone_depthMhz_{inputs.depth_mhz}_SKAobstime_{inputs.SKA_observation_time}_walker_{inputs.WalkerID:04d}_seed_{seed_indx}_slice_1.npy", x[:25, 25:])
    np.save(f"{inputs.saving_location}/lightcone_depthMhz_{inputs.depth_mhz}_SKAobstime_{inputs.SKA_observation_time}_walker_{inputs.WalkerID:04d}_seed_{seed_indx}_slice_2.npy", x[25:, :25])
    np.save(f"{inputs.saving_location}/lightcone_depthMhz_{inputs.depth_mhz}_SKAobstime_{inputs.SKA_observation_time}_walker_{inputs.WalkerID:04d}_seed_{seed_indx}_slice_3.npy", x[25:, 25:])
