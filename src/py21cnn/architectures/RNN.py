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
                FCsizes = [32, 16, 8],
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
    
    def conv2d_bn(self, x, filters, kernel_size, padding="valid", strides=(1, 1)):
        x = keras.layers.TimeDistributed(keras.layers.Conv2D(filters, 
                                                            kernel_size, 
                                                            strides=strides, 
                                                            padding=padding,
                                                            **self.AuxHP.ActivationFunction[1]))(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        return x
    def conv2d(self, x, filters, kernel_size, padding="valid", strides=(1, 1)):
        x = keras.layers.TimeDistributed(keras.layers.Conv2D(filters, 
                                                            kernel_size, 
                                                            strides=strides, 
                                                            padding=padding,
                                                            **self.AuxHP.ActivationFunction[1]))(x)
        return x

    def build(self):
        img_input = keras.layers.Input(shape=self.InputShape)
        #for the first layer apply batch norm after pooling
        x = self.conv2d(img_input, 64, (8, 8))
        x = keras.layers.TimeDistributed(keras.layers.MaxPooling2D())(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)

        x = self.conv2d_bn(x, 128, (4, 4))
        x = keras.layers.TimeDistributed(keras.layers.MaxPooling2D())(x)
        x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)
        x = keras.layers.TimeDistributed(self.DropoutLayer(self.AuxHP.Dropout))(x)
        x = keras.layers.TimeDistributed(keras.layers.Dense(128, **self.AuxHP.ActivationFunction[1]))(x)
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
