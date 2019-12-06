import sys
import importlib
import argparse
import src.py21cnn.utilities as utilities
# from src.py21cnn.architectures import *
import tensorflow as tf
from tensorflow import keras
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
print("inputs: ", inputs)
ModelClassObject = getattr(importlib.import_module(f'src.py21cnn.architectures.{inputs.model[0]}'), inputs.model[1])

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
                            [keras.layers.LeakyReLU(alpha=0.1), "leakyrelu"],
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

ModelClass = ModelClassObject(Data.shape, HP)
ModelClass.build()
utilities.run_model(model = ModelClass.model, 
                    Data = Data, 
                    AuxHP = HP,
                    inputs = inputs)