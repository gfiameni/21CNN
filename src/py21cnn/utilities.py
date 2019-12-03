import hashlib
import numpy as np
import tensorflow as tf
from tensorflow import keras

class Data:
    def __init__(
        self,
        filepath,
        dimensionality = 2,
        removed_average = True,
        normalized = True,
        Zmax = 30,
        filetype = 'float32',
        formatting = [],
        ):
        self.filepath = filepath
        self.dimensionality = dimensionality
        self.removed_average = removed_average
        self.normalized = normalized
        self.Zmax = Zmax
        self.filetype = filetype
        if len(formatting) == 0:
            default_formatting = ['clipped_-250_+50', 'NaN_removed']
            if self.dimensionality == 2:
                default_formatting.append('boxcar22')
            if self.dimensionality == 3:
                default_formatting.append('boxcar444')
            default_formatting.sort()
            self.formatting = default_formatting
        else:
            formatting.sort()
            self.formatting = formatting

    def __str__(self):
        S = f"dim:{self.dimensionality}__removed_average:{self.removed_average}__normalized:{self.normalized}__Zmax:{self.Zmax}__dtype:{self.filetype}"
        for i in self.formatting:
            S += f"__{i}"
        return S
    
    def hash(self):
        return hashlib.md5(self.__str__().encode()).hexdigest()

    def loadTVT(self, pTVT = [0.8, 0.1, 0.1]):
        self.X = {}
        self.Y = {}
        Hash = self.hash()
        for p, key in zip(pTVT, ['train', 'val', 'test']):
            self.X[key] = np.load(f"{self.filepath}X_{key}_{p:.2f}_{Hash}.npy")
            self.Y[key] = np.load(f"{self.filepath}Y_{key}_{p:.2f}_{Hash}.npy")
        # return self.X, self.Y


class AuxiliaryHyperparameters:
    def __init__(
        self,
        # Loss = {"instance": None, "name": "mse"},
        Loss = [None, "mse"],
        # Optimizer = {"instance": keras.optimizers.RMSprop(), "name": "RMSprop"},
        Optimizer = [keras.optimizers.RMSprop(), "RMSprop"],
        LearningRate = 0.01,
        # ActivationFunction = {"instance": keras.activations.relu(), "name": "relu"},
        ActivationFunction = [keras.activations.relu, "relu"],
        BatchNormalization = True,
        Dropout = 0.2,
        ReducingLR = False, 
        BatchSize = 20,
        Epochs = 200,
        ):
        self.Loss = Loss
        self.Optimizer = Optimizer
        self.LearningRate = LearningRate
        self.Optimizer[0].lr = LearningRate
        self.ActivationFunction = ActivationFunction
        self.BatchNormalization = BatchNormalization
        self.Dropout = Dropout
        self.ReducingLR = False
        self.BatchSize = BatchSize
        self.Epochs = Epochs

    def __str__(self):
        S = f"Loss:{self.Loss[1]}__Optimizer:{self.Optimizer[1]}__LR:{self.LearningRate:.10f}__Activation:{self.ActivationFunction[1]}"
        S += f"__BN:{self.BatchNormalization}__dropout:{self.Dropout:.2f}__reduceLR:{self.ReducingLR}__Batch:{self.BatchSize:05d}__Epochs:{self.Epochs:05d}"
        return S
    def hash(self):
        return hashlib.md5(self.__str__().encode()).hexdigest()


def coef_determination(y_true, y_pred):
        SS_res = keras.backend.sum(keras.backend.square( y_true-y_pred )) 
        SS_tot = keras.backend.sum(keras.backend.square( y_true - keras.backend.mean(y_true) ) )
#         loss = tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1)
        return ( 1 - SS_res/(SS_tot + keras.backend.epsilon()))

def run_model(Model, Data, AuxHP):
    pass