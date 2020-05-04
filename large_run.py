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
parser.add_argument('--pTVT', type=str, default = "0.8,0.1,0.1")

parser.add_argument('--data_location', type=str, default="data/")
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
inputs.model = inputs.model.split('.')
if len(inputs.model_type) == 0:
    inputs.model_type = inputs.model[0]
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
if ctx.inputs.tf == 1:
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
else:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if ctx.inputs.gpus > 1:
        hvd.init()
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # #assuming each node has one gpu, for other configurations, has to be properly modified
    # #gpus has only one member
    # gpu = gpus[0]
    # tf.config.experimental.set_memory_growth(gpu, True)
    # if ctx.inputs.gpus > 1:
    #     hvd.init()
    # tf.config.experimental.set_visible_devices(gpu, "GPU")

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
if ctx.inputs.simple_run == True:
    HP_dict = hyperparameters.HP_simple()
else:
    HP = hyperparameters.HP()
    HP_list = list(itertools.product(*HP.values()))
    HP_dict = dict(zip(HP.keys(), HP_list[ctx.inputs.HyperparameterIndex]))    

#correct number of epochs in multigpu training and LR
if ctx.inputs.gpus > 1:
    if ctx.inputs.LR_correction == True:
        HP_dict["LearningRate"] *= hvd.size()
    # print(f"IN define_model, HVD SIZE: {hvd.size()}")
    Epochs = ctx.inputs.epochs // hvd.size()
    MaxEpochs = ctx.inputs.max_epochs // hvd.size()
    # print(f"EPOCHS AND MAX EPOCHS: {ctx.HP.Epochs} {ctx.HP.MaxEpochs}")
else:
    Epochs = ctx.inputs.epochs
    MaxEpochs = ctx.inputs.max_epochs 
HP = utilities.AuxiliaryHyperparameters(
    model_name=f"{ctx.inputs.model[0]}_{ctx.inputs.model[1]}", 
    Epochs=Epochs, 
    MaxEpochs=MaxEpochs, 
    **HP_dict,
    )

print("HYPERPARAMETERS:", str(HP))
ctx.HP = HP

###############################################################################
#constructing TVT partitions of the data and assigning labels
###############################################################################
data_shape = ctx.inputs.X_shape[::-1] + (1,) if ctx.inputs.model_type == "RNN" else ctx.inputs.X_shape + (1,)
Data = utilities.LargeData(ctx, dimensionality = 3, shape = data_shape)

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