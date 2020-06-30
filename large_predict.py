###############################################################################
#define context, module with important variables
###############################################################################
from src.py21cnn import ctx
ctx.init()
###############################################################################
#parsing inputs
###############################################################################
import argparse
parser = argparse.ArgumentParser(prog = 'Large Database Model Run')

parser.add_argument('--simple_run', type=int, choices=[0, 1], default = 0)

parser.add_argument('--X_fstring', type=str, default = "lightcone_depthMhz_0_walker_{:04d}_slice_{:d}_seed_{:d}")
parser.add_argument('--X_shape', type=str, default="25,25,526")
parser.add_argument('--Y_filename', type=str, default = "NormalizedParams")
parser.add_argument('--N_walker', type=int, default=10000)
parser.add_argument('--N_slice', type=int, default=4)
parser.add_argument('--N_noise', type=int, default=10)
parser.add_argument('--noise_rolling', type=int, choices=[0, 1], default = 1)
parser.add_argument('--pTVT', type=str, default = "0.8,0.1,0.1")
parser.add_argument('--workers', type=int, default=24)
parser.add_argument('--load_all', type=int, default=0)
parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=2)

parser.add_argument('--data_location', type=str, default="data/")
parser.add_argument('--tfrecord_database', type=int, choices = [0, 1], default=0)

parser.add_argument('--saving_location', type=str, default="models/")
parser.add_argument('--logs_location', type=str, default="logs/")
parser.add_argument('--model', type=str, default="RNN.SummarySpace3D")
parser.add_argument('--model_type', type=str, default="")
parser.add_argument('--HyperparameterIndex', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--max_epochs', type=int, default=-1)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--LR_correction', type=int, choices=[0, 1], default=1)
parser.add_argument('--file_prefix', type=str, default="")
# parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--warmup', type=int, default=0)

parser.add_argument('--tf', type = int, choices = [1, 2], default = 1)

inputs = parser.parse_args()
inputs.LR_correction = bool(inputs.LR_correction)
inputs.simple_run = bool(inputs.simple_run)
inputs.noise_rolling = bool(inputs.noise_rolling)
inputs.load_all = bool(inputs.load_all)
inputs.tfrecord_database = bool(inputs.tfrecord_database)
inputs.model = inputs.model.split('.')
if len(inputs.model_type) == 0:
    inputs.model_type = inputs.model[0]
if inputs.tfrecord_database == True:
    if inputs.noise_rolling == False:
        raise ValueError("tfrecord_database is only compatible with noise rolling")
    if inputs.N_walker != 10000 or inputs.N_slice != 4:
        raise ValueError("for tfrecord database all walkers(10000) and slices(4) need to be chosen.")
    if inputs.pTVT != "0.8,0.1,0.1":
        raise ValueError("tfrecord databases fixes pTVT to 0.8,0.1,0.1")
inputs.pTVT = [float(i) for i in inputs.pTVT.split(',')]
inputs.X_shape = tuple([int(i) for i in inputs.X_shape.split(',')])
if inputs.max_epochs == -1:
    inputs.max_epochs = inputs.epochs
elif inputs.max_epochs < inputs.epochs:
    raise ValueError("epochs shouldn't be larger than max_epochs")

print("INPUTS:", inputs)
ctx.inputs = inputs

###############################################################################
#seting up GPUs
###############################################################################
import tensorflow as tf
if ctx.inputs.tf == 1:
    # #setting up GPU
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1. #setting the percentage of GPU usage
    config.gpu_options.visible_device_list = "0" #for picking only some devices
    config.gpu_options.allow_growth = True
    # config.log_device_placement=True
    # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    tf.compat.v1.enable_eager_execution(config=config)
else:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.enable_eager_execution()

#importing keras at the end, I had some issues if I import it before setting GPUs
from tensorflow import keras
keras.backend.set_image_data_format('channels_last')

ctx.main_process = True

###############################################################################
#seting hyperparameters
###############################################################################
import copy
import itertools
import sys
from src.py21cnn import utilities
from src.py21cnn import hyperparameters
if ctx.inputs.simple_run == True:
    HP_dict = hyperparameters.HP_simple()
else:
    HP = hyperparameters.HP()
    HP_list = list(itertools.product(*HP.values()))
    HP_dict = dict(zip(HP.keys(), HP_list[ctx.inputs.HyperparameterIndex]))    

#correct learning rate for multigpu run
if ctx.inputs.LR_correction == True:
        HP_dict["LearningRate"] *= ctx.inputs.gpus
        
HP = utilities.AuxiliaryHyperparameters(
    model_name=f"{ctx.inputs.model[0]}_{ctx.inputs.model[1]}", 
    Epochs=ctx.inputs.epochs, 
    MaxEpochs=ctx.inputs.max_epochs, 
    NoiseRolling=ctx.inputs.noise_rolling,
    **HP_dict,
    )

print("HYPERPARAMETERS:", str(HP))
ctx.HP = HP

###############################################################################
#constructing TVT partitions of the data and assigning labels
###############################################################################
data_shape = ctx.inputs.X_shape[::-1] + (1,) if ctx.inputs.model_type == "RNN" else ctx.inputs.X_shape + (1,)
data_class = utilities.Data_tfrecord if ctx.inputs.tfrecord_database == True else utilities.LargeData
Data = data_class(dimensionality = 3, shape = data_shape, load_all = ctx.inputs.load_all)

print("DATA:", str(Data))
ctx.Data = Data

ctx.filepath = f"{ctx.inputs.saving_location}{ctx.inputs.file_prefix}{ctx.inputs.model[0]}_{ctx.inputs.model[1]}_{ctx.HP.hash()}_{ctx.Data.hash()}"
ctx.logdir = f"{ctx.inputs.logs_location}{ctx.inputs.file_prefix}{ctx.inputs.model[0]}/{ctx.inputs.model[1]}/{ctx.Data.hash()}/{ctx.HP.hash()}"
print("FILENAME:", ctx.filepath)

###############################################################################
#predicting the model
###############################################################################
y_true, y_pred, r2_score = utilities.predict_large("best")

###############################################################################
#formatting and plotting
###############################################################################
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import matplotlib.gridspec as gridspec
import matplotlib.ticker as plticker
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter, AutoLocator
from matplotlib.gridspec import GridSpec
from matplotlib import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#plotting names and min max values of parameters
y_range = {
    "parameters": ["ZETA", "TVIR_MIN", "L_X", "NU_X_THRESH"], 
    "min": np.array([10.023063659667969, 4.000161170959473, 38.00041961669922, 100.13026428222656]), 
    "max": np.array([249.9544677734375, 5.998933792114258, 41.99989700317383, 1499.37109375]),
    "LatexNames": [
        r'\rm{\zeta}', 
        r'\rm{log_{10}(T_{vir})}', 
        r'\rm{log_{10}(L_X/SFR) }', 
        r'\rm{E_0}' 
        ],
    "LatexUnits": [
        '', 
        r'\rm{ [K] }', 
        r'\rm{ [erg\ s^{-1}\ M^{-1}_{\odot}\ yr] }', 
        r'\rm{ [eV] }' 
        ],
    }

# rescaling the actual data
minimum, maximum = ctx.Data.y_min, ctx.Data.y_max
print("BEFORE SCALING")
print("MIN MAX OF THE WHOLE DATA", minimum, maximum)
print("MIN MAX OF THE TRAIN DATA", np.amin(y_true, axis=0), np.amax(y_true, axis=0))
print("MIN MAX OF THE PRED DATA", np.amin(y_pred, axis=0), np.amax(y_pred, axis=0))
y_true = (y_true - minimum)/(maximum - minimum)
y_pred = (y_pred - minimum)/(maximum - minimum)
print('min max after scaling to 0-1')
print(np.amin(y_true, axis=0), np.amax(y_true, axis=0))
print(np.amin(y_pred, axis=0), np.amax(y_pred, axis=0))
print('min max after scaling to actual data, should be in y_range')
y_true = y_true * (y_range['max'] - y_range['min']) + y_range['min']
y_pred = y_pred * (y_range['max'] - y_range['min']) + y_range['min']
print(np.amin(y_true, axis=0), np.amax(y_true, axis=0))
print(np.amin(y_pred, axis=0), np.amax(y_pred, axis=0))

#plotting true vs predict
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
    im = ax_joint.imshow(hist, origin='low', vmin = 0, vmax = 1.5, interpolation=interp, cmap=plt.cm.jet, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax_joint.set_xlabel("$%s_{%s} \\,\\, True$"%(y_range["LatexNames"][i], y_range["LatexUnits"][i]), fontsize = 14)
    ax_joint.set_ylabel("$%s_{%s} \\,\\, Pred$"%(y_range["LatexNames"][i], y_range["LatexUnits"][i]), fontsize = 14)
    ax_joint.text(0.12, 0.95, "$R^2 = %.4f$"%(r2_score[i]), horizontalalignment='center', verticalalignment='center', transform=ax_joint.transAxes, fontsize = 16, color='white')

    ax_marg_x.hlines(0, minmax[0], minmax[1], colors='k')
    ax_marg_x.errorbar(centers, delta['pred'], deviation['pred'])
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    ax_marg_x.set_ylabel("$\\epsilon_{%s}$"%(y_range["LatexUnits"][i]), fontsize = 14)
    ax_marg_x.set_xlim(minmax[0], minmax[1])
#     ax_marg_x.set_title("$R^2 = %.4f$"%(r2_score[i]) , fontsize = 16)
    ax_marg_x.tick_params(direction='in')


    ax_marg_y.vlines(0, minmax[0], minmax[1], colors='k')
    ax_marg_y.errorbar(delta['true'], centers, xerr = deviation['true'])
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    ax_marg_y.set_xlabel("$\\epsilon_{%s}$"%(y_range["LatexUnits"][i]), fontsize = 14)
    ax_marg_y.set_ylim(minmax[0], minmax[1])
    ax_marg_y.tick_params(direction='in')


    
#     fig.suptitle(y_range["LatexNames"][i])
    plt.tight_layout()
    cbaxes = inset_axes(ax_joint, width="3%", height="30%", loc=4, borderpad=0.35)
    cbar = fig.colorbar(im, cax=cbaxes, ticks=[0, 1], orientation='vertical')
    cbaxes.tick_params(color = 'white')

#     cbar = fig.colorbar(im, ticks=[0, 0.5, 1, 1.5], pad=0.2)
    cbar.set_label("$\\log_{10} (N_{pix} + 1)$", fontsize = 14, labelpad=-40, color='white')
    plt.savefig(f'{ctx.filepath}_true_vs_pred_best_{i}.pdf')
#     plt.show()


#plotting 
import csv

with open(ctx.filepath + '.log', 'r') as f:
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
    
plt.figure(figsize = (6, 4))
plt.plot(logs['epoch'], np.log10(logs['loss']), label = 'train')
plt.plot(logs['epoch'], np.log10(logs['val_loss']), label = 'validation')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("$\\log_{10} L$")
# plt.yticks([-1., -1.5, -2, -2.5])
plt.grid()
plt.tight_layout()
plt.savefig(f'{ctx.filepath}_loss_best.pdf')
# plt.show()