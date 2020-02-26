import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as plticker
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter, AutoLocator
from matplotlib.gridspec import GridSpec
from matplotlib import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np

from time import time
# import pyemma.utils_functions as utils
import sys
import json
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import argparse
parser = argparse.ArgumentParser(prog = 'Plot ICs')
parser.add_argument('--data_location', type=str, default="../data/")
parser.add_argument('--pics_location', type=str, default="../pics/")
parser.add_argument('--filename', type=str, default="")
inputs = parser.parse_args()

import csv

pics_filepath = f"{inputs.pics_location}{inputs.filename}"

train = np.zeros((10, 100))
val = np.zeros((10, 100))

plt.figure(figsize = (6, 4))
for i in range(10):
    filepath = f"{inputs.data_location}{i}{inputs.filename}"
    with open(filepath + '.log', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        rows = [row for row in reader if row]
        headings = rows[0] # get headings
        logs = {}
        for row in rows[1:]:
            # append the dataitem to the end of the dictionary entry
            # set the default value of [] if this key has not been seen
            for col_header, data_column in zip(headings, row):
                logs.setdefault(col_header, []).append(float(data_column))
    for key in logs.keys():
        logs[key] = np.array(logs[key])
    if i == 0:
        label_train = "train"
        label_val = "validation"
    else:
        label_train = None
        label_val = None
    plt.plot(logs['epoch'] + 1, np.log10(logs['loss']), 'r-', label = label_train, linewidth=0.5)
    plt.plot(logs['epoch'] + 1, np.log10(logs['val_loss']), 'b-', label = label_val, linewidth = 0.5)

    train[i] = logs['loss']
    val[i] = logs['val_loss']

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("$\log_{10} L$")
# plt.yticks([-1., -1.5, -2, -2.5])
plt.grid()
plt.tight_layout()
plt.savefig(pics_filepath + f'_ICloss.pdf')
# plt.show()


stats_train = {
    'mean': np.mean(train, axis = 0),
    'std' : np.std(train, axis = 0)
}
stats_val = {
    'mean': np.mean(val, axis = 0),
    'std' : np.std(val, axis = 0)
}

plt.figure(figsize = (6, 4))
plt.plot(logs['epoch'] + 1, np.log10(stats_train['mean']), 'r-', label = 'train')
plt.plot(logs['epoch'] + 1, np.log10(stats_val['mean']), 'b-', label = 'validation')
plt.fill_between(logs['epoch'] + 1, np.log10(stats_train['mean'] - 3*stats_train['std']), np.log10(stats_train['mean'] + 3*stats_train['std']), color='red', alpha=0.5)
plt.fill_between(logs['epoch'] + 1, np.log10(stats_val['mean'] - 3*stats_val['std']), np.log10(stats_val['mean'] + 3*stats_val['std']), color='blue', alpha=0.5)

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("$\log_{10} L$")
# plt.yticks([-1., -1.5, -2, -2.5])
plt.grid()
plt.tight_layout()
plt.savefig(pics_filepath + f'_ICloss_std.pdf')
# plt.show()