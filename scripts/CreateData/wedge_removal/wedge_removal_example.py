import argparse
parser = argparse.ArgumentParser(prog = 'Run Model')
parser.add_argument('--WalkerID', type=int, choices=range(10000), default=0)
parser.add_argument('--uv_filename', type=str, default='uv_final.npy')
parser.add_argument('--W_filepath', type=str, default='')
parser.add_argument('--saving_location', type=str, default='')
parser.add_argument('--averages_fstring', type=str, default='averages_{}_float32.npy')
parser.add_argument('--BoxesPath', type=str, default="/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/LightConeBoxes")
parser.add_argument('--ParametersPath', type=str, default="/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/GridPositions")
parser.add_argument('--depth_mhz', type=int, default = 0) # if depth==0 calculate it from cube, else fix it to given value
# parser.add_argument('--uv_treshold', type=int, default = 1) # taking in account only baselines which were visited uv_treshold amount of times 
parser.add_argument('--SKA_observation_time', type=int, default = 1000)
# inputs = parser.parse_args("--WalkerID 9999 --uv_filepath ../data/uv_Steven_15.npy --saving_location ../DatabaseTest --BoxesPath ../DatabaseTest --averages_fstring ../DatabaseTest/averages_{}_float32.npy".split(" "))
inputs = parser.parse_args()

import numpy as np
import sys
import os
from src.py21cnn.database import DatabaseUtils
from src.py21cnn.formatting import Filters
import json

import matplotlib.pyplot as plt
import matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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
Box = Box.astype(np.float32)

uv = np.load(inputs.uv_filename)
uv_bool = (uv < 1)

import tools21cm as t2c
N_ant = 512 #or 513?

t2c.const.set_hubble_h(0.678)
t2c.const.set_omega_matter(0.308)
t2c.const.set_omega_baryon(0.048425)
t2c.const.set_omega_lambda(0.692)
t2c.const.set_ns(0.968)
t2c.const.set_sigma_8(0.815)

d0 = t2c.cosmology.z_to_cdist(float(Redshifts[0]))
cdist = np.array(range(Box.shape[-1] + 1))*1.5 + d0
redshifts = t2c.cosmology.cdist_to_z(cdist)
redshifts_mean = (redshifts[:-1] + redshifts[1:]) / 2


def noise(depth_mhz, seed_index):
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
        finalBox.append(noise)
    finalBox = np.moveaxis(np.array(finalBox), 0, -1)
#     finalBox[uv_bool] = 0
    return finalBox

Noise = noise(inputs.depth_mhz, seed_index = 0).astype(np.float32)

def simple_sliding(W_bool, Box):
    assert(W_bool.shape[-1] == Box.shape[-1])
    
    Box_final = np.empty(Box.shape, dtype = np.float32)
#     Noise = noise(inputs.depth_mhz, seed_index = 0).astype(np.float32)
    Box_inv = np.fft.fft2(Box, axes=(0, 1)) + Noise
    Box_inv[uv_bool] = 0
    Box_inv = np.fft.fft(Box_inv, axis = -1)
    
    for i in range(Box.shape[-1])
        print(i, end=' ')
        Box_final[..., i] = np.real(np.fft.ifftn(Box_inv * W_bool[i, ...]))[..., i]
    return Box_final


def sliding(W_bool, Box, chunk_length = 200, blackman = True):
    assert(W_bool.shape[-1] == chunk_length)
    Box_final = np.empty(Box.shape, dtype = np.float32)
#     Noise = noise(inputs.depth_mhz, seed_index = 0).astype(np.float32)
    Box_uv = np.fft.fft2(Box, axes=(0, 1)) + Noise
    Box_uv[uv_bool] = 0
    
    BM = np.blackman(chunk_length)[np.newaxis, np.newaxis, :]
    
    for i in len(redshifts_mean):
        if i < chunk_length // 2 or i > Box.shape[-1] - chunk_length // 2:
            continue
        elif i == chunk_length // 2:
            t_box = Box_uv[..., i - chunk_length // 2: i + chunk_length // 2]
            Box_final[..., 0:chunk_length // 2] = np.real(np.fft.ifftn(np.fft.fft(t_box, axis = -1) * W_bool[i, ...]))[..., 0:chunk_length // 2]
        elif i == Box.shape[-1] - chunk_length // 2:
            t_box = Box_uv[..., i - chunk_length // 2: i + chunk_length // 2]
            Box_final[..., -chunk_length // 2:] = np.real(np.fft.ifftn(np.fft.fft(t_box, axis = -1) * W_bool[i, ...]))[..., -chunk_length // 2:]
        else:
            t_box = Box_uv[..., i - chunk_length // 2: i + chunk_length // 2]
            if blackman == True:
                t_box *= BM
            Box_final[..., i] = np.real(np.fft.ifftn(np.fft.fft(t_box, axis = -1) * W_bool[i, ...]))[chunk_length // 2]
    return Box_final

def slicing(W_bool, Box, chunk_length = 200):
    assert(W_bool.shape[-1] == chunk_length)
    slices = Box.shape[-1] // chunk_length
    Box_final = np.empty(slices * chunk_length, dtype = np.float32)
#     Noise = noise(inputs.depth_mhz, seed_index = 0).astype(np.float32)
    Box_uv = np.fft.fft2(Box, axes=(0, 1)) + Noise
    Box_uv[uv_bool] = 0
    
    for i in range(slices):
        t_box = Box_uv[..., i * chunk_length: (i + 1) * chunk_length]
        W_index = (i + 1) * slices if i != slices - 1 else -1
        Box_final[..., i * chunk_length: (i + 1) * chunk_length] = np.real(np.fft.ifftn(np.fft.fft(t_box, axis = -1) * W_bool[W_index, ...]))
    return Box_final

def plotting(box, box_cleaned, filename):
    fig, ax =plt.subplots(2, 1, figsize=(25, 3*2))
    im = ax[0].imshow(box[0], vmin = -1e2, vmax = 1e2)
    ax[0].set_xticks(np.array(range(9)) * 250)
    ax[0].set_xticklabels([ f"{i:.1f}" for i in t2c.cosmology.cdist_to_z(np.array(range(9)) * 250 * 1.5 + d0)])
    ax[0].set_yticks([])
    plt.colorbar(im, ax = ax[0], fraction=0.005, pad=0.005)
    ax[0].set_title("signal + noise", fontsize=16)

    fig, ax =plt.subplots(2, 1, figsize=(25, 3*2))
    im = ax[1].imshow(box_cleaned[0], vmin = -1e2, vmax = 1e2)
    ax[1].set_xticks(np.array(range(9)) * 250)
    ax[1].set_xticklabels([ f"{i:.1f}" for i in t2c.cosmology.cdist_to_z(np.array(range(9)) * 250 * 1.5 + d0)])
    ax[1].set_yticks([])
    plt.colorbar(im, ax = ax[1], fraction=0.005, pad=0.005)
    ax[1].set_title("signal + noise + wedge removal", fontsize=16)
    # plt.suptitle(f"uv lightcone", fontsize=20)
    plt.savefig(filename)
#     plt.show()

W_bool = np.load(f"{inputs.W_filepath}W_2107.npy")

Box_final = simple_sliding(W_bool, Box)
plotting(Box, Box_final, "simple_sliding.pdf")


W_bool = np.load(f"{inputs.W_filepath}W_200.npy")

Box_final = sliding(W_bool, Box, blackman = True)
plotting(Box, Box_final, "sliding_blackman.pdf")

Box_final = sliding(W_bool, Box, blackman = False)
plotting(Box, Box_final, "sliding.pdf")

Box_final = slicing(W_bool, Box)
plotting(Box, Box_final, "slicing.pdf")