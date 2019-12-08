import tensorflow as tf
import keras
# #setting up GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1. #setting the percentage of GPU usage
config.gpu_options.visible_device_list = "0" #for picking only some devices
# config.gpu_options.allow_growth = True
config.log_device_placement=True
keras.backend.set_session(tf.Session(config=config))
keras.backend.set_image_data_format('channels_last')

import src.py21cnn.utilities as utilities
import numpy as np
import sys
import importlib
import argparse
import itertools


parser = argparse.ArgumentParser(prog = 'Run Model')
parser.add_argument('--dimensionality', type=int, choices=[2, 3], default=3)
parser.add_argument('--removed_average', type=int, choices=[0, 1], default=1)
parser.add_argument('--Zmax', type=int, default=30)
parser.add_argument('--data_location', type=str, default="/scratch/../../")
parser.add_argument('--saving_location', type=str, default="/scratch/../../")
parser.add_argument('--model', type=str, default="RNN.SummarySpace3D")
parser.add_argument('--HyperparameterIndex', type=int, choices=range(768), default=0)
parser.add_argument('--epochs', type=int, default=200)

inputs = parser.parse_args()
inputs.removed_average = bool(inputs.removed_average)
inputs.model = inputs.model.split('.')
print("INPUTS: ", inputs)
ModelClassObject = getattr(importlib.import_module(f'src.py21cnn.architectures.{inputs.model[0]}'), inputs.model[1])

def leakyrelu(x):
    return keras.layers.relu(x, alpha=0.1)

HyP = {}
HyP["Loss"] = [[None, "mse"]]
HyP["Epochs"] = [inputs.epochs]
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
HyP["LearningRate"] = [0.1, 0.01, 0.001, 0.0001]
HyP["Dropout"] = [0, 0.3]
HyP["ReducingLR"] = [True]
HyP["BatchSize"] = [20, 200]
HyP["BatchNormalization"] = [True, False]
HyP["ActivationFunction"] = [
                            [keras.activations.relu, "relu"],
                            # [keras.layers.LeakyReLU(alpha=0.1), "leakyrelu"],
                            [leakyrelu, "leakyrelu"]
                            [keras.activations.elu, "elu"],
                            # [keras.activations.selu, "selu"],
                            [keras.activations.exponential, "exponential"],
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
print("DATA", str(Data))

ModelClass = ModelClassObject(Data.shape, HP)
ModelClass.build()
utilities.run_model(model = ModelClass.model, 
                    Data = Data, 
                    AuxHP = HP,
                    inputs = inputs)