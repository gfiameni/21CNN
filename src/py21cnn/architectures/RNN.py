import tensorflow as tf
# import keras
from tensorflow import keras
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

class SummarySpace3D:
    def __init__(self, 
                InputShape, 
                # Data, 
                AuxiliaryHP, 
                # RNNLayer = keras.layers.CuDNNLSTM,
                RNNLayer = CuDNNLSTM,
                RNNsizes = [128, 64, 64],
                FCsizes = [32, 32, 16, 8],
                ):
        self.InputShape = InputShape
        # self.Data = Data
        self.AuxHP = AuxiliaryHP
        self.RNNLayer = RNNLayer
        self.RNNsizes = RNNsizes
        self.FCsizes = FCsizes
        if self.AuxHP.ActivationFunction[0] == "selu":
            self.DropoutLayer = keras.layers.AlphaDropout
        else:
            self.DropoutLayer = keras.layers.Dropout
    
    # def conv2d_bn(self, x, filters, kernel_size, padding="valid", strides=(1, 1)):
    #     x = keras.layers.TimeDistributed(keras.layers.Conv2D(filters, 
    #                                                         kernel_size, 
    #                                                         strides=strides, 
    #                                                         padding=padding,
    #                                                         **self.AuxHP.ActivationFunction[1]))(x)
    #     if self.AuxHP.BatchNormalization == True:
    #         x = keras.layers.BatchNormalization()(x)
    #     return x
    # def conv2d(self, x, filters, kernel_size, padding="valid", strides=(1, 1)):
    #     x = keras.layers.TimeDistributed(keras.layers.Conv2D(filters, 
    #                                                         kernel_size, 
    #                                                         strides=strides, 
    #                                                         padding=padding,
    #                                                         **self.AuxHP.ActivationFunction[1]))(x)
    #     return x

    def build(self):
        img_input = keras.layers.Input(shape=self.InputShape)

        x = keras.layers.TimeDistributed(keras.layers.Conv2D(64, (8, 8), **self.AuxHP.ActivationFunction[1]), name = 'conv1')(img_input)
        x = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(), name = 'maxpool1')(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.TimeDistributed(keras.layers.Conv2D(128, (4, 4), **self.AuxHP.ActivationFunction[1]), name = 'conv2')(x)
        x = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(), name = 'maxpool2')(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.TimeDistributed(keras.layers.Flatten(), name = 'flatten')(x)
        x = keras.layers.TimeDistributed(self.DropoutLayer(self.AuxHP.Dropout), name = 'dropout')(x)
        x = keras.layers.TimeDistributed(keras.layers.Dense(128, **self.AuxHP.ActivationFunction[1]), name = 'summary space')(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)

        for i in self.RNNsizes:
            x = self.RNNLayer(i, return_sequences=True)(x)
            if self.AuxHP.BatchNormalization == True:
                x = keras.layers.BatchNormalization()(x)
        #first dense layer is stil RNN
        x = self.RNNLayer(self.FCsizes[0], return_sequences=False)(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)

        # x = self.DropoutLayer(self.AuxHP.Dropout)(x)

        for i in self.FCsizes[1:]:
            x = keras.layers.Dense(i, **self.AuxHP.ActivationFunction[1])(x)
            # if self.AuxHP.BatchNormalization == True:
            #     x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Dense(4)(x)

        self.model = keras.models.Model(inputs = img_input, outputs = x, name= "RNN_SummarySpace3D")
        print(self.model.summary())
        return self.model

class SummarySpace2D:
    def __init__(self, 
                InputShape, 
                # Data, 
                AuxiliaryHP, 
                # RNNLayer = keras.layers.CuDNNLSTM,
                RNNLayer = CuDNNLSTM,
                RNNsizes = [128, 64, 64],
                FCsizes = [32, 32, 16, 8],
                ):
        self.InputShape = InputShape
        # self.Data = Data
        self.AuxHP = AuxiliaryHP
        self.RNNLayer = RNNLayer
        self.RNNsizes = RNNsizes
        self.FCsizes = FCsizes
        if self.AuxHP.ActivationFunction[0] == "selu":
            self.DropoutLayer = keras.layers.AlphaDropout
        else:
            self.DropoutLayer = keras.layers.Dropout
    

    def build(self):
        img_input = keras.layers.Input(shape=self.InputShape)

        x = keras.layers.TimeDistributed(keras.layers.Conv1D(16, 8, **self.AuxHP.ActivationFunction[1]), name = 'conv1')(img_input)
        x = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(), name = 'maxpool1')(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.TimeDistributed(keras.layers.Conv1D(64, 4, **self.AuxHP.ActivationFunction[1]), name = 'conv2')(x)
        x = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(), name = 'maxpool2')(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.TimeDistributed(keras.layers.Flatten(), name = 'flatten')(x)
        x = keras.layers.TimeDistributed(self.DropoutLayer(self.AuxHP.Dropout), name = 'dropout')(x)
        x = keras.layers.TimeDistributed(keras.layers.Dense(128, **self.AuxHP.ActivationFunction[1]), name = 'summary space')(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)     

        for i in self.RNNsizes:
            x = self.RNNLayer(i, return_sequences=True)(x)
            if self.AuxHP.BatchNormalization == True:
                x = keras.layers.BatchNormalization()(x)
        #first dense layer is stil RNN
        x = self.RNNLayer(self.FCsizes[0], return_sequences=False)(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)

        # x = self.DropoutLayer(self.AuxHP.Dropout)(x)

        for i in self.FCsizes[1:]:
            x = keras.layers.Dense(i, **self.AuxHP.ActivationFunction[1])(x)
            # if self.AuxHP.BatchNormalization == True:
            #     x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Dense(4)(x)

        self.model = keras.models.Model(inputs = img_input, outputs = x, name= "RNN_SummarySpace2D")
        print(self.model.summary())
        return self.model

class ConvRNN3D:
    def __init__(self, 
                InputShape, 
                # Data, 
                AuxiliaryHP, 
                # RNNLayer = keras.layers.CuDNNLSTM,
                RNNLayer = CuDNNLSTM,
                RNNsizes = [128, 64, 64],
                FCsizes = [32, 32, 16, 8],
                ):
        self.InputShape = InputShape
        # self.Data = Data
        self.AuxHP = AuxiliaryHP
        self.RNNLayer = RNNLayer
        self.RNNsizes = RNNsizes
        self.FCsizes = FCsizes
        if self.AuxHP.ActivationFunction[0] == "selu":
            self.DropoutLayer = keras.layers.AlphaDropout
        else:
            self.DropoutLayer = keras.layers.Dropout
    
    def build(self):
        img_input = keras.layers.Input(shape=self.InputShape)

        x = keras.layers.ConvLSTM2D(filters=32, kernel_size=(8, 8), return_sequences=True)(img_input)
        x = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(), name='pool_1')(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ConvLSTM2D(filters=64, kernel_size=(4, 4), return_sequences=True)(x)
        x = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(), name='pool_2')(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.TimeDistributed(keras.layers.Flatten(), name='flatten')(x)
        x = keras.layers.TimeDistributed(self.DropoutLayer(self.AuxHP.Dropout), name='dropout')(x)
        x = self.RNNLayer(128, return_sequences=True)(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)

        for i in self.RNNsizes:
            x = self.RNNLayer(i, return_sequences=True)(x)
            if self.AuxHP.BatchNormalization == True:
                x = keras.layers.BatchNormalization()(x)
        #first dense layer is stil RNN
        x = self.RNNLayer(self.FCsizes[0], return_sequences=False)(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)

        # x = self.DropoutLayer(self.AuxHP.Dropout)(x)

        for i in self.FCsizes[1:]:
            x = keras.layers.Dense(i, **self.AuxHP.ActivationFunction[1])(x)
            # not doing batch norm on last layers

        x = keras.layers.Dense(4)(x)

        self.model = keras.models.Model(inputs = img_input, outputs = x, name= "RNN_ConvRNN3D")
        print(self.model.summary())
        return self.model


class Hybrid3D:
    def __init__(self,
                InputShape, 
                # Data, 
                AuxiliaryHP, 
                # RNNLayer = keras.layers.CuDNNLSTM,
                RNNLayer = CuDNNLSTM,
                RNNsizes = [128, 64, 64],
                FCsizes = [32, 32, 16, 8],
                ):
        self.InputShape = InputShape
        # self.Data = Data
        self.AuxHP = AuxiliaryHP
        self.RNNLayer = RNNLayer
        self.RNNsizes = RNNsizes
        self.FCsizes = FCsizes
        if self.AuxHP.ActivationFunction[0] == "selu":
            self.DropoutLayer = keras.layers.AlphaDropout
        else:
            self.DropoutLayer = keras.layers.Dropout        

    def build(self):
        img_input = keras.layers.Input(shape=self.InputShape)
        
        x = keras.layers.Conv3D(128, (8, 8, 8), **self.AuxHP.ActivationFunction[1])(img_input)
        x = keras.layers.MaxPooling3D(pool_size=(4, 2, 2), strides=(4, 2, 2))(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv3D(128, (1, 4, 4), **self.AuxHP.ActivationFunction[1])(x)
        x = keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.TimeDistributed(keras.layers.Flatten(), name='flatten')(x)
        x = keras.layers.TimeDistributed(self.DropoutLayer(self.AuxHP.Dropout), name='dropout')(x)
        x = keras.layers.TimeDistributed(keras.layers.Dense(128, **self.AuxHP.ActivationFunction[1]), name = 'dense')(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)        

        for i in self.RNNsizes:
            x = self.RNNLayer(i, return_sequences=True)(x)
            if self.AuxHP.BatchNormalization == True:
                x = keras.layers.BatchNormalization()(x)
        #first dense layer is stil RNN
        x = self.RNNLayer(self.FCsizes[0], return_sequences=False)(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)

        # x = self.DropoutLayer(self.AuxHP.Dropout)(x)

        for i in self.FCsizes[1:]:
            x = keras.layers.Dense(i, **self.AuxHP.ActivationFunction[1])(x)
            # if self.AuxHP.BatchNormalization == True:
            #     x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Dense(4)(x)

        self.model = keras.models.Model(inputs = img_input, outputs = x, name= "RNN_Hybrid3D")
        print(self.model.summary())
        return self.model        


