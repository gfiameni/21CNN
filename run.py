import argparse
###############################################################################
#define context variable, Namespace with all important objects
###############################################################################
ctx = argparse.Namespace()
###############################################################################
#parsing inputs
###############################################################################
parser = argparse.ArgumentParser(prog = 'Run Model')
parser.add_argument('--dimensionality', type=int, choices=[2, 3], default=3)
parser.add_argument('--removed_average', type=int, choices=[0, 1], default=1)
parser.add_argument('--Zmax', type=int, default=30)
parser.add_argument('--data_location', type=str, default="data/")
parser.add_argument('--saving_location', type=str, default="models/")
parser.add_argument('--logs_location', type=str, default="logs/")
parser.add_argument('--model', type=str, default="RNN.SummarySpace3D")
parser.add_argument('--model_type', type=str, default="")
parser.add_argument('--HyperparameterIndex', type=int, choices=range(576), default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--max_epochs', type=int, default=-1)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--LR_correction', type=int, choices=[0, 1], default=1)
parser.add_argument('--file_prefix', type=str, default="")
# parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--warmup', type=int, default=0)

inputs = parser.parse_args()
inputs.removed_average = bool(inputs.removed_average)
inputs.LR_correction = bool(inputs.LR_correction)
inputs.model = inputs.model.split('.')
if len(inputs.model_type) == 0:
    inputs.model_type = inputs.model[0]
if inputs.max_epochs == -1:
    inputs.max_epochs = inputs.epochs
elif inputs.max_epochs > inputs.epochs:
    raise ValueError("max_epochs shouldn't be larger than epochs")

print("INPUTS:", inputs)
ctx.inputs = inputs

###############################################################################
#seting up GPUs
###############################################################################
import tensorflow as tf
import horovod.tensorflow.keras as hvd
# tf.compat.v1.enable_eager_execution()
if ctx.inputs.gpus == 1:
    # #setting up GPU
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1. #setting the percentage of GPU usage
    config.gpu_options.visible_device_list = "0" #for picking only some devices
    config.gpu_options.allow_growth = True
    # config.log_device_placement=True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    # tf.compat.v1.enable_eager_execution(config=config)
elif ctx.inputs.gpus > 1:
    #init Horovod
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    # tf.compat.v1.enable_eager_execution(config=config)
else:
    raise ValueError('number of gpus shoud be > 0')
#importing keras at the end, I had some issues if I import it before setting GPUs
from tensorflow import keras
keras.backend.set_image_data_format('channels_last')

if ctx.inputs.gpus > 1:
    print("HVD.SIZE", hvd.size())

if ctx.inputs.gpus > 1:
    ctx.main_process = True if hvd.rank() == 0 else False
else:
    ctx.main_process = True

###############################################################################
#seting hyperparameters
###############################################################################
import copy
import itertools
import sys
from src.py21cnn import utilities
from src.py21cnn import hyperparameters
HP = hyperparameters.HP()
HP_list = list(itertools.product(*HP.values()))
HP_dict = dict(zip(HP.keys(), HP_list[ctx.inputs.HyperparameterIndex]))
if ctx.inputs.LR_correction == True and ctx.inputs.gpus > 1:
    HP_dict["LearningRate"] *= hvd.size()
HP = utilities.AuxiliaryHyperparameters(
    model_name=f"{ctx.inputs.model[0]}_{ctx.inputs.model[1]}", 
    Epochs=ctx.inputs.epochs, 
    MaxEpochs=ctx.inputs.max_epochs, 
    **HP_dict,
    )

print("HYPERPARAMETERS:", str(HP))
ctx.HP = HP

###############################################################################
#constructing TVT partitions of the data and assigning labels
###############################################################################
Data = utilities.Data(
    filepath=ctx.inputs.data_location, 
    dimensionality=ctx.inputs.dimensionality, 
    removed_average=ctx.inputs.removed_average, 
    Zmax=ctx.inputs.Zmax,
    )
Data.loadTVT(model_type=ctx.inputs.model_type)

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

utilities.run_model()