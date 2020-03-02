import argparse
parser = argparse.ArgumentParser(prog = 'Run Model')
parser.add_argument('--WalkerID', type=int, choices=range(10000), default=0)
parser.add_argument('--uv_filepath', type=str, default='uv.npy')
parser.add_argument('--saving_location', type=str, default='')
parser.add_argument('--averages_fstring', type=str, default='averages_{}_float32.npy')
parser.add_argument('--BoxesPath', type=str, default="/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/LightConeBoxes")
parser.add_argument('--ParametersPath', type=str, default="/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/GridPositions")
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
# print(cdist)
redshifts = t2c.cosmology.cdist_to_z(cdist)
redshifts_mean = (redshifts[:-1] + redshifts[1:]) / 2
# print(redshifts)
# print(redshifts.shape)
# print(redshifts_mean.shape)

finalBox = []
for i in range(Box.shape[-1]):
    noise = t2c.noise_model.noise_map(ncells=200, 
                                      z=redshifts_mean[i], 
                                      depth_mhz=t2c.cosmology.z_to_nu(redshifts[i]) - t2c.cosmology.z_to_nu(redshifts[i+1]),
                                      boxsize=300,
                                      uv_map=uv[..., i],
                                      N_ant=N_ant)
    noise = t2c.telescope_functions.jansky_2_kelvin(noise, redshifts_mean[i])
    x = np.fft.fft2(Box[..., i]) + noise
    x[uv[..., i]==0] = 0
    # print (x.shape)
    finalBox.append(np.real(np.fft.ifft2(x)))
    
finalBox = np.moveaxis(np.array(finalBox, dtype=np.float32), 0, -1)
print(finalBox.shape)
finalBox, _  = t2c.smoothing.smooth_lightcone(finalBox, z_array=redshifts_mean, box_size_mpc=300)
np.save(f"{inputs.saving_location}/lightcone_{inputs.WalkerID:04d}.npy", finalBox[::4, ::4, ::4])
