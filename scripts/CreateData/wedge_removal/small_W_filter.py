import argparse
parser = argparse.ArgumentParser(prog = 'construct W filter')
parser.add_argument('--saving_location', type=str, default='')
parser.add_argument('--dimensions', type=str, default='200,200,201')
parser.add_argument('--wedge_correction', type=float, default=5)

inputs = parser.parse_args()
inputs.dimensions = tuple([int(x) for x in inputs.dimensions.split(",")])

import numpy as np
import cupy as cp

import tools21cm as t2c
N_ant = 512 #or 513?

t2c.const.set_hubble_h(0.678)
t2c.const.set_omega_matter(0.308)
t2c.const.set_omega_baryon(0.048425)
t2c.const.set_omega_lambda(0.692)
t2c.const.set_ns(0.968)
t2c.const.set_sigma_8(0.815)

Redshifts = ['006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42503', \
            '013.25424', '015.51874', '018.36856', '022.02434', '026.82138', '033.28927', '034.50984']
Box_shape = (inputs.dimensions[0], inputs.dimensions[1], 2107)

d0 = t2c.cosmology.z_to_cdist(float(Redshifts[0]))
cdist = np.array(range(Box_shape[-1] + 1))*1.5 + d0
redshifts = t2c.cosmology.cdist_to_z(cdist)
redshifts_mean = (redshifts[:-1] + redshifts[1:]) / 2

k0, k1, k2 = cp.fft.fftfreq(inputs.dimensions[0], d=1.5), cp.fft.fftfreq(inputs.dimensions[1], d=1.5), cp.fft.fftfreq(inputs.dimensions[2], d=1.5)
delta_k = k0[1] - k0[0]
k_cube = cp.meshgrid(k0, k1, k2)

BM = cp.abs(cp.fft.fft(cp.blackman(inputs.dimensions[2])))**2
BM = BM / cp.amax(BM)
BM_smoothing = delta_k * (cp.where(BM <= 1e-10)[0][0] - 1)

from scipy.integrate import quadrature
def one_over_E(z):
    return 1 / np.sqrt(t2c.const.Omega0*(1.+z)**3+t2c.const.lam)
def multiplicative_factor(z):
    return 1 / one_over_E(z) / (1+z) * quadrature(one_over_E, 0, z)[0]
multiplicative_fact = cp.array([multiplicative_factor(z) for z in redshifts_mean]).astype(np.float32)

W_bool = np.empty((Box_shape[-1],) + inputs.dimensions, dtype = bool)
for i in range(len(multiplicative_fact)):
    W = k_cube[2] / (cp.sqrt(k_cube[0]**2 + k_cube[1]**2) * (multiplicative_fact[i] / inputs.wedge_correction) + BM_smoothing)
    W_bool[i, ...] = cp.logical_or(W < -1., W > 1.).get()

np.save(f"{inputs.saving_location}W_{inputs.dimensions[-1]}", W_bool)