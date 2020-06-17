import time
tic = time.time()
def timing():
    global tic
    toc = time.time()
    print(toc-tic)
    tic = toc
    
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
import cupy as cp
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
print("move box to GPU")
Box = cp.asarray(Box)
print("clipping large values")
cp.clip(Box, deltaTmin, deltaTmax, out=Box)

BoxAverage = np.load(inputs.averages_fstring.format(inputs.WalkerID))
print("removing large Z for average")
BoxAverage = Filters.RemoveLargeZ(BoxAverage, database, Z=Zmax)
BoxAverage = cp.asarray(BoxAverage)

Box -= BoxAverage
Box = Box.astype(np.float32)

uv = np.load(inputs.uv_filename).astype(np.float32)
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

Noise = noise(inputs.depth_mhz, seed_index = 0).astype(np.complex64)
#move everything to GPU
Noise = cp.asarray(Noise)
uv = cp.asarray(uv)
uv_bool = cp.asarray(uv_bool)

def simple_sliding(W_bool, Box):
    assert(W_bool.shape[-1] == Box.shape[-1])
    Box_final = cp.empty(Box.shape, dtype = np.float32)
#     Noise = noise(inputs.depth_mhz, seed_index = 0).astype(np.float32)
    Box_inv = cp.fft.fft2(Box, axes=(0, 1)) + Noise
    Box_inv[uv_bool] = 0
    Box_inv = cp.fft.fft(Box_inv, axis = -1)
    
    for i in range(Box.shape[-1]):
        # box_inv = np.copy(Box_inv)
        # box_inv[W_bool[i, ...]] = 0
        # Box_final[..., i] = np.real(np.fft.ifftn(box_inv))[..., i]
        w = cp.asarray(W_bool[i, ...]) # as W_bool is too large for GPU
        Box_final[..., i] = cp.real(cp.fft.ifftn(Box_inv * w))[..., i]
    return Box_final


def sliding(W_bool, Box, chunk_length = 200, blackman = True):
    assert(W_bool.shape[-1] == chunk_length)
    Box_final = cp.empty(Box.shape, dtype = np.float32)
#     Noise = noise(inputs.depth_mhz, seed_index = 0).astype(np.float32)
    Box_uv = cp.fft.fft2(Box, axes=(0, 1)) + Noise
    Box_uv[uv_bool] = 0
    
    BM = cp.blackman(chunk_length)[cp.newaxis, cp.newaxis, :]
    
    for i in len(redshifts_mean):
        w = cp.asarray(W_bool[i, ...]) #should put on GPU the whole W in final version
        if i < chunk_length // 2 or i > Box.shape[-1] - chunk_length // 2:
            continue
        elif i == chunk_length // 2:
            t_box = Box_uv[..., i - chunk_length // 2: i + chunk_length // 2]
            Box_final[..., 0:chunk_length // 2] = cp.real(cp.fft.ifftn(cp.fft.fft(t_box, axis = -1) * w))[..., 0:chunk_length // 2]
        elif i == Box.shape[-1] - chunk_length // 2:
            t_box = Box_uv[..., i - chunk_length // 2: i + chunk_length // 2]
            Box_final[..., -chunk_length // 2:] = cp.real(cp.fft.ifftn(cp.fft.fft(t_box, axis = -1) * w))[..., -chunk_length // 2:]
        else:
            t_box = Box_uv[..., i - chunk_length // 2: i + chunk_length // 2]
            if blackman == True:
                t_box *= BM
            Box_final[..., i] = cp.real(cp.fft.ifftn(cp.fft.fft(t_box, axis = -1) * w))[chunk_length // 2]
    return Box_final

def slicing(W_bool, Box, chunk_length = 200):
    assert(W_bool.shape[-1] == chunk_length)
    slices = Box.shape[-1] // chunk_length
    Box_final = cp.empty(slices * chunk_length, dtype = np.float32)
#     Noise = noise(inputs.depth_mhz, seed_index = 0).astype(np.float32)
    Box_uv = cp.fft.fft2(Box, axes=(0, 1)) + Noise
    Box_uv[uv_bool] = 0
    
    for i in range(slices):
        t_box = Box_uv[..., i * chunk_length: (i + 1) * chunk_length]
        W_index = (i + 1) * slices if i != slices - 1 else -1
        w = cp.asarray(W_bool[W_index, ...])
        Box_final[..., i * chunk_length: (i + 1) * chunk_length] = cp.real(cp.fft.ifftn(cp.fft.fft(t_box, axis = -1) * w))
    return Box_final

def plotting(box, box_cleaned, box_n_noise, filename):
    box = box.get()
    box_cleaned = box_cleaned.get()
    box_n_noise = box_n_noise.get()
    fig, ax =plt.subplots(3, 1, figsize=(25, 3*3))
    im = ax[0].imshow(box[0], vmin = -1e2, vmax = 1e2)
    ax[0].set_xticks(np.array(range(9)) * 250)
    ax[0].set_xticklabels([ f"{i:.1f}" for i in t2c.cosmology.cdist_to_z(np.array(range(9)) * 250 * 1.5 + d0)])
    ax[0].set_yticks([])
    plt.colorbar(im, ax = ax[0], fraction=0.005, pad=0.005)
    ax[0].set_title("signal + noise", fontsize=16)

    im = ax[1].imshow(box_n_noise[0], vmin = -1e2, vmax = 1e2)
    ax[1].set_xticks(np.array(range(9)) * 250)
    ax[1].set_xticklabels([ f"{i:.1f}" for i in t2c.cosmology.cdist_to_z(np.array(range(9)) * 250 * 1.5 + d0)])
    ax[1].set_yticks([])
    plt.colorbar(im, ax = ax[1], fraction=0.005, pad=0.005)
    ax[1].set_title("signal + noise", fontsize=16)

    im = ax[2].imshow(box_cleaned[0], vmin = -1e2, vmax = 1e2)
    ax[2].set_xticks(np.array(range(9)) * 250)
    ax[2].set_xticklabels([ f"{i:.1f}" for i in t2c.cosmology.cdist_to_z(np.array(range(9)) * 250 * 1.5 + d0)])
    ax[2].set_yticks([])
    plt.colorbar(im, ax = ax[1], fraction=0.005, pad=0.005)
    ax[2].set_title("signal + noise + wedge_removal", fontsize=16)
    # plt.suptitle(f"uv lightcone", fontsize=20)
    plt.savefig(filename)
#     plt.show()

#only for plotting purposes
box_n_noise = cp.fft.fft2(Box, axes=(0, 1)) + Noise
box_n_noise[uv_bool] = 0
box_n_noise = cp.real(cp.fft.ifft2(box_n_noise, axes=(0, 1)))

timing()
print("LOADING LARGE W")
W_bool = np.load(f"{inputs.W_filepath}W_2107.npy")
timing()
print("CALCULATING simple_sliding")
Box_final = simple_sliding(W_bool, Box)
plotting(Box, Box_final, box_n_noise, f"{inputs.saving_location}simple_sliding_{inputs.WalkerID:04d}.pdf")
timing()
print("LOADING SMALL W")
W_bool = np.load(f"{inputs.W_filepath}W_200.npy")
timing()
print("CALCULATING sliding_blackman")
Box_final = sliding(W_bool, Box, blackman = True)
plotting(Box, Box_final, box_n_noise, f"{inputs.saving_location}sliding_blackman_{inputs.WalkerID:04d}.pdf")
timing()
print("CALCULATING sliding")
Box_final = sliding(W_bool, Box, blackman = False)
plotting(Box, Box_final, box_n_noise, f"{inputs.saving_location}sliding_{inputs.WalkerID:04d}.pdf")
timing()
print("CALCULATING slicing")
Box_final = slicing(W_bool, Box)
plotting(Box, Box_final, box_n_noise, f"{inputs.saving_location}slicing_{inputs.WalkerID:04d}.pdf")
timing()