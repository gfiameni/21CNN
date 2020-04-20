###############################################################################
#parsing inputs
###############################################################################
import argparse
parser = argparse.ArgumentParser(prog = 'Large Database Model Run')
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
parser.add_argument('--HyperparameterIndex', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--max_epochs', type=int, default=-1)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--multi_gpu_correction', type=str, choices=["none", "batch_size", "learning_rate"], default="none")
parser.add_argument('--file_prefix', type=str, default="")
parser.add_argument('--patience', type=int, default=10)

inputs = parser.parse_args()
inputs.model = inputs.model.split('.')
inputs.pTVT = [float(i) for i in inputs.pTVT.split(',')]
inputs.X_shape = tuple([float(i) for i in inputs.X_shape.split(',')])
if inputs.max_epochs == -1:
    inputs.max_epochs = inputs.epochs
elif inputs.max_epochs > inputs.epochs:
    raise ValueError("max_epochs shouldn't be larger than epochs")
print("INPUTS:", inputs)

###############################################################################
#constructing TVT partitions of the data and assigning labels
###############################################################################
import numpy as np
from src.py21cnn.formatting import Filters
permutation = Filters.constructIndexArray(inputs.N_walker, *inputs.pTVT, 1312)
print(permutation)

Y = np.load(f"{inputs.data_location}{inputs.Y_filename}.npy")
partition = {
    "train": [], 
    "validation": [], 
    "test": []}
labels = {}
keys = list(partition.keys())
for walker in range(inputs.N_walker):
    for s in range(inputs.N_slice):
        for seed in range(inputs.N_noise):
            partition[keys[permutation[walker]]].append(inputs.X_fstring.format(walker, s, seed))
            labels[inputs.X_fstring.format(walker, s, seed)] = Y[walker]
print(partition)

###############################################################################
#seting up GPUs
###############################################################################
import tensorflow as tf
import horovod.tensorflow.keras as hvd
# tf.compat.v1.enable_eager_execution()
if inputs.gpus == 1:
    # #setting up GPU
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1. #setting the percentage of GPU usage
    config.gpu_options.visible_device_list = "0" #for picking only some devices
    config.gpu_options.allow_growth = True
    # config.log_device_placement=True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    # tf.compat.v1.enable_eager_execution(config=config)
elif inputs.gpus > 1:
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

###############################################################################
#importing model
###############################################################################
import importlib
import src.py21cnn.utilities as utilities
ModelClassObject = getattr(importlib.import_module(f'src.py21cnn.architectures.{inputs.model[0]}'), inputs.model[1])


###############################################################################
#seting hyperparameters
###############################################################################
import copy
import itertools
import sys
from src.py21cnn import hyperparameters
HP = hyperparameters.HP(inputs)
HP_list = list(itertools.product(*HP.values()))
HP_dict = dict(zip(HP.keys(), HP_list[inputs.HyperparameterIndex]))
HP = utilities.AuxiliaryHyperparameters(**HP_dict)
#creating HP dict for TensorBoard with only HP that are changing and only human-readable information
HP_TensorBoard = {
    "Model": f"{inputs.model[0]}_{inputs.model[1]}",
    "LearningRate": HP_dict["LearningRate"],
    "Dropout": HP_dict["Dropout"],
    "BatchSize": HP_dict["BatchSize"],
    "BatchNormalization": HP_dict["BatchNormalization"],
    "Optimizer": HP_dict["Optimizer"][1],
    "ActivationFunction": HP_dict["ActivationFunction"][0],
}

# Data = utilities.Data(filepath=inputs.data_location, 
#                       dimensionality=inputs.dimensionality, 
#                       removed_average=inputs.removed_average, 
#                       Zmax=inputs.Zmax)
# Data.loadTVT(model_type=inputs.model[0])
print("HYPERPARAMETERS:", str(HP))
# print("DATA:", str(Data))
if inputs.gpus > 1:
    print("HVD.SIZE", hvd.size())



###############################################################################
#building and running the model
###############################################################################
# ModelClass = ModelClassObject(Data.shape, HP)
# ModelClass.build()
# if inputs.gpus == 1:
#     utilities.run_model(model = ModelClass.model, 
#                         Data = Data, 
#                         AuxHP = HP,
#                         HP_TensorBoard = HP_TensorBoard,
#                         inputs = inputs)
# else:
#     #corrections for multigpu
#     AuxHP = copy.deepcopy(HP)
#     if inputs.multi_gpu_correction == 2:
#         AuxHP.Optimizer[2]["lr"] *= hvd.size()
#     elif inputs.multi_gpu_correction == 1:
#         AuxHP.BatchSize //= hvd.size()
#     AuxHP.Epochs //= hvd.size()
#     AuxHP.MaxEpochs //=hvd.size()
#     print("BEFORE RUN AuxHP: ", str(AuxHP))
#     print("BEFORE RUN HP: ", str(HP))

#     utilities.run_multigpu_model(model = ModelClass.model, 
#                                 Data = Data, 
#                                 AuxHP = AuxHP,
#                                 HP = HP,
#                                 HP_TensorBoard = HP_TensorBoard,
#                                 inputs = inputs,
#                                 # hvd = hvd,
#                                 )

# # Parameters
# params = {'dim': (32,32,32),
#           'batch_size': 64,
#           'n_classes': 6,
#           'n_channels': 1,
#           'shuffle': True}

# # Generators
# training_generator = DataGenerator(partition['train'], labels, **params)
# validation_generator = DataGenerator(partition['validation'], labels, **params)


# model.compile()

# # Train model on dataset
# model.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,
#                     use_multiprocessing=True,
#                     workers=6)

