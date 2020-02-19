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

from scipy.stats import gaussian_kde
from scipy.stats import kde
from astropy.modeling import models, fitting

import numpy as np

from time import time
# import pyemma.utils_functions as utils
import sys
import json
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import argparse
parser = argparse.ArgumentParser(prog = 'Run Model')
parser.add_argument('--dimensionality', type=int, choices=[2, 3], default=3)
parser.add_argument('--removed_average', type=int, choices=[0, 1], default=1)
parser.add_argument('--Zmax', type=int, default=30)
parser.add_argument('--data_location', type=str, default="../data/")
parser.add_argument('--saving_location', type=str, default="../models/")
parser.add_argument('--logs_location', type=str, default="../logs/")
parser.add_argument('--model', type=str, default="RNN.SummarySpace3D")
parser.add_argument('--HyperparameterIndex', type=int, choices=range(576), default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--multi_gpu_correction', type=int, choices=[0, 1, 2], default=0, help="0-none, 1-batch_size, 2-learning_rate")
parser.add_argument('--file_prefix', type=str, default="")

inputs = parser.parse_args(['--model', 'CNN.basic2D', '--dimensionality', '2', '--HyperparameterIndex', '237'])
inputs.removed_average = bool(inputs.removed_average)
inputs.model = inputs.model.split('.')
print("INPUTS:", inputs)


import tensorflow as tf
# #setting up GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1. #setting the percentage of GPU usage
config.gpu_options.visible_device_list = "0" #for picking only some devices
config.gpu_options.allow_growth = True
# config.log_device_placement=True
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
tf.compat.v1.enable_eager_execution(config=config)

from tensorflow import keras
keras.backend.set_image_data_format('channels_last')

import copy
import itertools
import sys
import importlib
import numpy as np

import src.py21cnn.utilities as utilities
ModelClassObject = getattr(importlib.import_module(f'src.py21cnn.architectures.{inputs.model[0]}'), inputs.model[1])

def leakyrelu(x):
    return keras.activations.relu(x, alpha=0.1)

HyP = {}
HyP["Loss"] = [[None, "mse"]]
HyP["Epochs"] = [inputs.epochs]
HyP["BatchSize"] = [20, 100]
HyP["LearningRate"] = [0.01, 0.001, 0.0001]
HyP["Dropout"] = [0.2, 0.5]
HyP["ReducingLR"] = [True]
HyP["BatchNormalization"] = [True, False]
HyP["Optimizer"] = [
                    [keras.optimizers.RMSprop, "RMSprop", {}],
                    [keras.optimizers.SGD, "SGD", {}],
                    [keras.optimizers.SGD, "Momentum", {"momentum":0.9, "nesterov":True}],
                    # [keras.optimizers.Adadelta, "Adadelta", {}],
                    # [keras.optimizers.Adagrad, "Adagrad", {}],
                    [keras.optimizers.Adam, "Adam", {}],
                    # [keras.optimizers.Adam, "Adam", {"amsgrad":True}],
                    [keras.optimizers.Adamax, "Adamax", {}],
                    [keras.optimizers.Nadam, "Nadam", {}],
                    ]
HyP["ActivationFunction"] = [
                            ["relu", {"activation": keras.activations.relu, "kernel_initializer": keras.initializers.he_uniform()}],
                            # [keras.layers.LeakyReLU(alpha=0.1), "leakyrelu"],
                            ["leakyrelu", {"activation": leakyrelu, "kernel_initializer": keras.initializers.he_uniform()}],
                            ["elu", {"activation": keras.activations.elu, "kernel_initializer": keras.initializers.he_uniform()}],
                            ["selu", {"activation": keras.activations.selu, "kernel_initializer": keras.initializers.lecun_normal()}],
                            # [keras.activations.exponential, "exponential"],
                            # [keras.activations.tanh, "tanh"],
                            ]


HyP_list = list(itertools.product(*HyP.values()))
HP_dict = dict(zip(HyP.keys(), HyP_list[inputs.HyperparameterIndex]))
HP = utilities.AuxiliaryHyperparameters(**HP_dict)
#creating HP dict for TensorBoard with only HP that are changing and only human-readable information
HP_TensorBoard = {}
HP_TensorBoard["Model"] = f"{inputs.model[0]}_{inputs.model[1]}"
HP_TensorBoard["LearningRate"] = HP_dict["LearningRate"]
HP_TensorBoard["Dropout"] = HP_dict["Dropout"]
HP_TensorBoard["BatchSize"] = HP_dict["BatchSize"]
HP_TensorBoard["BatchNormalization"] = HP_dict["BatchNormalization"]
HP_TensorBoard["Optimizer"] = HP_dict["Optimizer"][1]
HP_TensorBoard["ActivationFunction"] = HP_dict["ActivationFunction"][0]

Data = utilities.Data(filepath=inputs.data_location, 
                      dimensionality=inputs.dimensionality, 
                      removed_average=inputs.removed_average, 
                      Zmax=inputs.Zmax)


def R2(y_true, y_pred):
        SS_res = keras.backend.sum(keras.backend.square(y_true-y_pred)) 
        SS_tot = keras.backend.sum(keras.backend.square(y_true - keras.backend.mean(y_true, axis=0)))
        return (1 - SS_res/(SS_tot + keras.backend.epsilon()))
    
    
y_true = np.load(f"{Data.filepath}Y_test_0.10_{Data.hash()}.npy")

filepath = f"{inputs.saving_location}{inputs.file_prefix}{inputs.model[0]}_{inputs.model[1]}_{HP.hash()}_{Data.hash()}"
custom_obj = {}
custom_obj["R2"] = R2
# custom_obj["TimeHistory"] = TimeHistory
#if activation is leakyrelu add to custom_obj
if HP.ActivationFunction[0] == "leakyrelu":
    custom_obj[HP.ActivationFunction[0]] = HP.ActivationFunction[1]["activation"]
model = keras.models.load_model(f"{filepath}_best.hdf5", custom_objects=custom_obj)
model.summary()

X_test = np.load(f"{Data.filepath}X_test_0.10_{Data.hash()}.npy")
X_test = X_test[..., np.newaxis]

y_pred = model.predict(X_test)


with open(f"{inputs.data_location}databaseParams_min_max.txt") as f:
    y_range = json.load(f)
y_range["LatexNames"] = [r'\rm{\zeta}', 
                         r'\rm{log_{10}(T_{vir})}', 
                         r'\rm{log_{10}(L_X/SFR) }', 
                         r'\rm{E_0}' ]
y_range["LatexUnits"] = ['', 
                         r'\rm{ [K] }', 
                         r'\rm{ [erg\ s^{-1}\ M^{-1}_{\odot}\ yr] }', 
                         r'\rm{ [eV] }' ]

def rescale(y_true, y_pred, y_range, filepath, pTVT = [0.8, 0.1, 0.1]):
    Y = {}
    for p, key in zip(pTVT, ['train', 'val', 'test']):
        Y[key] = np.load(f"{filepath}Y_{key}_{p:.2f}_{Data.hash()}.npy")
    y_tot = np.concatenate((Y['train'], Y['val'], Y['test']), axis=0)
    print(y_tot.shape)
    minimum, maximum = np.amin(y_tot, axis=0), np.amax(y_tot, axis = 0)
    print(minimum, maximum)
    y_true = (y_true - minimum)/(maximum - minimum)
    y_pred = (y_pred - minimum)/(maximum - minimum)
    print('should be 0-1')
    print(np.amin(y_true, axis=0), np.amax(y_true, axis=0))
    print(np.amin(y_pred, axis=0), np.amax(y_pred, axis=0))
    y_range['min'] = np.array(y_range['min'])
    y_range['max'] = np.array(y_range['max'])
    print('should be in y_range')
    y_true = y_true * (y_range['max'] - y_range['min']) + y_range['min']
    y_pred = y_pred * (y_range['max'] - y_range['min']) + y_range['min']
    print(np.amin(y_true, axis=0), np.amax(y_true, axis=0))
    print(np.amin(y_pred, axis=0), np.amax(y_pred, axis=0))
    return y_true, y_pred


y_true, y_pred = rescale(y_true, y_pred, y_range, inputs.data_location)


def R2( out_model, Param):
    return 1 - ( (out_model - Param)**2).sum(axis=0) / ((out_model - out_model.mean(axis=0) )**2).sum(axis=0) 


r2_score = np.zeros(4)
for i in range(4):
    r2_score[i] = R2(y_pred[:, i], y_true[:, i])
print(r2_score)








nbins = 100
for i in range(4):
    #defining plot
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4,4)
    ax_joint = fig.add_subplot(gs[1:4,0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[1:4,3])

    #calculating main plots and marginalizations
    x = y_true[:, i]
    y = y_pred[:, i]
#     print(x, y)
    minmax = [y_range['min'][i], y_range['max'][i]]
    hist, xedges, yedges = np.histogram2d(x, y, range = [minmax, minmax], bins=nbins)
    centers = (xedges[:-1] + xedges[1:]) / 2
    hist = np.log10(1 + hist.T)

    marginalize_true = [[] for i in range(nbins)]
    marginalize_pred = [[] for i in range(nbins)]
    hist_delta = (minmax[1] - minmax[0]) / nbins
    for true, pred in zip(x, y):
        index_true = int((true - minmax[0])/hist_delta)
#         if index_true >= nbins:
#             index_true = nbins-1
        index_pred = int((pred - minmax[0])/hist_delta)
#         if index_pred >= nbins:
#             index_pred = nbins-1
#         print(true, index_true)
#         print(pred, index_pred)
        if index_pred < nbins:
            marginalize_true[index_pred].append(true - pred)
        if index_true < nbins:
            marginalize_pred[index_true].append(pred - true)   
#     print(marginalize_pred[nbins//2])
    delta = {'true': [np.mean(k) if len(k) > 0 else 0 for k in marginalize_true], 'pred': [np.mean(k) if len(k) > 0 else 0 for k in marginalize_pred]}
    deviation = {'true': [np.std(k, ddof = 1) if len(k) > 0 else 0 for k in marginalize_true], 'pred': [np.std(k, ddof = 1) if len(k) > 0 else 0 for k in marginalize_pred]}

    #plotting axes
    interp = 'gaussian'
#     interp = None
    im = ax_joint.imshow(hist, origin='low', vmin = 0, vmax = 2.5, interpolation=interp, cmap=plt.cm.jet, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax_joint.set_xlabel("$%s_{%s} \,\, True$"%(y_range["LatexNames"][i], y_range["LatexUnits"][i]), fontsize = 14)
    ax_joint.set_ylabel("$%s_{%s} \,\, Pred$"%(y_range["LatexNames"][i], y_range["LatexUnits"][i]), fontsize = 14)
    ax_joint.text(0.13, 0.95, "$R^2 = %.4f$"%(r2_score[i]), horizontalalignment='center', verticalalignment='center', transform=ax_joint.transAxes, fontsize = 16, color='white')

    ax_marg_x.hlines(0, minmax[0], minmax[1], colors='k')
    ax_marg_x.errorbar(centers, delta['pred'], deviation['pred'])
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    ax_marg_x.set_ylabel("$\epsilon_{%s}$"%(y_range["LatexUnits"][i]), fontsize = 14)
    ax_marg_x.set_xlim(minmax[0], minmax[1])
#     ax_marg_x.set_title("$R^2 = %.4f$"%(r2_score[i]) , fontsize = 16)
    ax_marg_y.vlines(0, minmax[0], minmax[1], colors='k')
    ax_marg_y.errorbar(delta['true'], centers, xerr = deviation['true'])
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    ax_marg_y.set_xlabel("$\epsilon_{%s}$"%(y_range["LatexUnits"][i]), fontsize = 14)
    ax_marg_y.set_ylim(minmax[0], minmax[1])

    
#     fig.suptitle(y_range["LatexNames"][i])
#     plt.tight_layout()
    cbar = fig.colorbar(im, ticks=[0, 1, 2], pad=0.2)
    cbar.set_label("$\log_{10} (N_{pix} + 1)$", fontsize = 14)
    plt.show()