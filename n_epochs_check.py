import tensorflow as tf
# import keras
from tensorflow import keras

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
parser.add_argument('--saving_location', type=str, default="/scratch/../../")
parser.add_argument('--model', type=str, default="RNN.SummarySpace3D")
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--file_prefix', type=str, default="")


inputs = parser.parse_args()
inputs.removed_average = bool(inputs.removed_average)
inputs.model = inputs.model.split('.')
print("INPUTS:", inputs)
# ModelClassObject = getattr(importlib.import_module(f'src.py21cnn.architectures.{inputs.model[0]}'), inputs.model[1])

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

Data = utilities.Data(filepath=inputs.data_location, 
                      dimensionality=inputs.dimensionality, 
                      removed_average=inputs.removed_average, 
                      Zmax=inputs.Zmax)
print("DATA:", str(Data))

HyP_list = list(itertools.product(*HyP.values()))
for i, h in enumerate(HyP_list):
    HP_dict = dict(zip(HyP.keys(), h))
    HP = utilities.AuxiliaryHyperparameters(**HP_dict)
    filepath = f"{inputs.saving_location}{inputs.file_prefix}{inputs.model[0]}_{inputs.model[1]}_{HP.hash()}_{Data.hash()}"
    with open(f"{filepath}_time.txt") as f:
            number_of_epochs_trained = len(f.readlines())
    if number_of_epochs_trained < inputs.epochs:
        print(i, str(HP))
