###############################################################################
#define context, module with important variables
###############################################################################
from src.py21cnn import ctx
ctx.init()


###############################################################################
#parsing inputs
###############################################################################
import argparse

from __future__ import absolute_import, division, print_function
# from timeutil import timeme

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
parser.add_argument('--tensorboard', type=int, choices=[0, 1], default=1)
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

# parser.add_argument('--tf', type = int, choices = [1, 2], default = 2)

inputs = parser.parse_args()
inputs.LR_correction = bool(inputs.LR_correction)
inputs.simple_run = bool(inputs.simple_run)
inputs.noise_rolling = bool(inputs.noise_rolling)
inputs.load_all = bool(inputs.load_all)
inputs.tfrecord_database = bool(inputs.tfrecord_database)
inputs.tensorboard = bool(inputs.tensorboard)
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
import horovod.tensorflow.keras as hvd
gpus = tf.config.experimental.list_physical_devices('GPU')
if ctx.inputs.gpus > 1:
    hvd.init()
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

#importing keras at the end, I had some issues if I import it before setting GPUs
from tensorflow import keras
keras.backend.set_image_data_format('channels_last')

if ctx.inputs.gpus > 1:
    print("HVD.SIZE", hvd.size())

if ctx.inputs.gpus > 1:
    ctx.main_process = True if hvd.rank() == 0 else False
else:
    ctx.main_process = True
    
# Configure to use XLA compiler 
USE_XLA = True
tf.config.optimizer.set_jit(USE_XLA)

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
Data.load()

print("DATA:", str(Data))
ctx.Data = Data

ctx.filepath = f"{ctx.inputs.saving_location}{ctx.inputs.file_prefix}{ctx.inputs.model[0]}_{ctx.inputs.model[1]}_{ctx.HP.hash()}_{ctx.Data.hash()}"
ctx.logdir = f"{ctx.inputs.logs_location}{ctx.inputs.file_prefix}{ctx.inputs.model[0]}/{ctx.inputs.model[1]}/{ctx.Data.hash()}/{ctx.HP.hash()}"

###############################################################################
#building and running the model
###############################################################################
import importlib
ModelClassObject = getattr(importlib.import_module(f'src.py21cnn.architectures.{ctx.inputs.model[0]}'), ctx.inputs.model[1])
ModelClass = ModelClassObject(ctx.Data.shape, HP)
ModelClass.build()

ctx.model = ModelClass.model

utilities.run_large_model()
