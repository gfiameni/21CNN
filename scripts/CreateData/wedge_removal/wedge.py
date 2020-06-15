import numpy as np
import tools21cm as t2c

Box_shape = (200, 200, 2107)
Redshifts = ['006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42503', \
            '013.25424', '015.51874', '018.36856', '022.02434', '026.82138', '033.28927', '034.50984']

t2c.const.set_hubble_h(0.678)
t2c.const.set_omega_matter(0.308)
t2c.const.set_omega_baryon(0.048425)
t2c.const.set_omega_lambda(0.692)
t2c.const.set_ns(0.968)
t2c.const.set_sigma_8(0.815)

d0 = t2c.cosmology.z_to_cdist(float(Redshifts[0]))
cdist = np.array(range(Box_shape[-1] + 1))*1.5 + d0
redshifts = t2c.cosmology.cdist_to_z(cdist)
redshifts_mean = (redshifts[:-1] + redshifts[1:]) / 2

from scipy.integrate import quadrature

def one_over_E(z):
    return 1 / np.sqrt(t2c.const.Omega0*(1.+z)**3+t2c.const.lam)
def multiplicative_factor(z):
    return 1 / one_over_E(z) / (1+z) * quadrature(one_over_E, 0, z)[0]

print("calculating muliplicative factors")
multiplicative_fact = np.array([multiplicative_factor(z) for z in redshifts_mean])
multiplicative_fact = multiplicative_fact[..., np.newaxis, np.newaxis, np.newaxis].astype(np.float32)

#for the total cube
k_perp = np.fft.fftfreq(Box_shape[0], d=1.5)
k_parallel = np.fft.fftfreq(Box_shape[-1], d=1.5)
k_cube = np.meshgrid(k_perp, k_perp, k_parallel)

print("calculating for 2107 cube")
W = k_cube[2] / np.sqrt(k_cube[0]**2 + k_cube[1]**2)
W = np.broadcast_to(W[np.newaxis, ...], (Box_shape[-1], *Box_shape)).astype(np.float32)
W = W / multiplicative_fact
W = ~((W >= -1.) * (W <= 1.)) # ~ is negative a of bool array, saved as bool to save memory
np.save("W_2107.npy", W)

#for 200 slice
print("calculating for 200 slice")
k = np.fft.fftfreq(200, d=1.5)
k_cube = np.meshgrid(k, k, k)

W = k_cube[2] / np.sqrt(k_cube[0]**2 + k_cube[1]**2)
W = np.broadcast_to(W[np.newaxis, ...], (Box_shape[-1],) + (200,)*3).astype(np.float32)
W = W / multiplicative_fact
W = ~((W >= -1.) * (W <= 1.))
np.save("W_200.npy", W)
