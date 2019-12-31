import argparse

parser = argparse.ArgumentParser(prog = 'Run Model')
parser.add_argument('--dimensionality', type=int, choices=[2, 3], default=3)
parser.add_argument('--removed_average', type=int, choices=[0, 1], default=1)
parser.add_argument('--Zmax', type=int, default=30)
parser.add_argument('--data_location', type=str, default="data/")
parser.add_argument('--saving_location', type=str, default="models/")
parser.add_argument('--model', type=str, default="RNN.SummarySpace3D")
parser.add_argument('--HyperparameterIndex', type=int, choices=range(576), default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--gpus', type=int, default=1)
# parser.add_argument('--multi_gpu_correction', type=int, choices=[0, 1, 2], default=0, help="0-none, 1-batch_size, 2-learning_rate")
parser.add_argument('--file_prefix', type=str, default="")

inputs = parser.parse_args()
inputs.removed_average = bool(inputs.removed_average)
inputs.model = inputs.model.split('.')
print("INPUTS:", inputs)

import copy
import itertools
import sys
import importlib
import numpy as np
import tensorflow as tf
# import keras
from tensorflow import keras
import horovod.tensorflow.keras as hvd

if inputs.gpus == 1:
    # #setting up GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1. #setting the percentage of GPU usage
    config.gpu_options.visible_device_list = "0" #for picking only some devices
    config.gpu_options.allow_growth = True
    # config.log_device_placement=True
    keras.backend.set_session(tf.Session(config=config))
elif inputs.gpus > 1:
    #init Horovod
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    keras.backend.set_session(tf.Session(config=config))
else:
    raise ValueError('number of gpus shoud be > 0')
keras.backend.set_image_data_format('channels_last')

import src.py21cnn.utilities as utilities
ModelClassObject = getattr(importlib.import_module(f'src.py21cnn.architectures.{inputs.model[0]}'), inputs.model[1])

def leakyrelu(x):
    return keras.activations.relu(x, alpha=0.1)

HyP = {}
HyP["Loss"] = [[None, "mse"]]
HyP["Epochs"] = [inputs.epochs]
HyP["LearningRate"] = [0.01, 0.001, 0.0001]
HyP["Dropout"] = [0.2, 0.5]
HyP["ReducingLR"] = [True]
HyP["BatchSize"] = [20, 100]
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

Data = utilities.Data(filepath=inputs.data_location, 
                      dimensionality=inputs.dimensionality, 
                      removed_average=inputs.removed_average, 
                      Zmax=inputs.Zmax)
Data.loadTVT(model_type=inputs.model[0])

print("HYPERPARAMETERS:", str(HP))
print("DATA:", str(Data))
print("HVD.SIZE", hvd.size())

ModelClass = ModelClassObject(Data.shape, HP)
ModelClass.build()
if inputs.gpus == 1:
    utilities.run_model(model = ModelClass.model, 
                        Data = Data, 
                        AuxHP = HP,
                        inputs = inputs)
else:

    #corrections for multigpu
    AuxHP = copy.deepcopy(HP)
    if AuxHP.BatchSize == 20:
        OptimizerCorrection = hvd.size()
        BatchSizeCorrection = 1
    else:
        OptimizerCorrection = hvd.size() // 5
        BatchSizeCorrection = 5
    AuxHP.Optimizer[2]["lr"] *= OptimizerCorrection
    AuxHP.BatchSize //= BatchSizeCorrection

    AuxHP.Epochs //= hvd.size()
    print("BEFORE RUN AuxHP: ", str(AuxHP))
    print("BEFORE RUN HP: ", str(HP))

    utilities.run_multigpu_model(model = ModelClass.model, 
                                Data = Data, 
                                AuxHP = AuxHP,
                                HP = HP,
                                inputs = inputs,
                                hvd = hvd,
                                )