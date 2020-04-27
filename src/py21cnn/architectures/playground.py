import tensorflow as tf
# import keras
from tensorflow import keras
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

class SummarySpace3D_simple:
    def __init__(self, 
                InputShape, 
                AuxiliaryHP, 
                RNNLayer = CuDNNLSTM,
                RNNsize = 16,
                even_simpler = True,
                ):
        self.InputShape = InputShape
        self.AuxHP = AuxiliaryHP
        self.RNNLayer = RNNLayer
        self.RNNsize = RNNsize
        self.even_simpler = even_simpler
        if self.AuxHP.ActivationFunction[0] == "selu":
            self.DropoutLayer = keras.layers.AlphaDropout
        else:
            self.DropoutLayer = keras.layers.Dropout

    def build(self):
        img_input = keras.layers.Input(shape=self.InputShape)

        x = keras.layers.TimeDistributed(keras.layers.Conv2D(64, (8, 8), **self.AuxHP.ActivationFunction[1]), name = 'conv1')(img_input)
        if self.even_simpler == True:
            x = keras.layers.TimeDistributed(keras.layers.GlobalMaxPooling2D(), name = 'maxpool')(x)
        else:
            x1 = keras.layers.TimeDistributed(keras.layers.GlobalMaxPooling2D(), name = 'maxpool')(x)
            x2 = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D(), name = 'avgpool')(x)
            x = keras.layers.concatenate([x1, x2], axis=-1)


        if self.AuxHP.BatchNormalization == True: x = keras.layers.BatchNormalization()(x)
        x = keras.layers.TimeDistributed(self.DropoutLayer(self.AuxHP.Dropout), name = 'dropout')(x)
        x = self.RNNLayer(self.RNNsize, return_sequences=False)(x)
        x = keras.layers.Dense(4)(x)

        self.model = keras.models.Model(inputs = img_input, outputs = x, name= self.AuxHP.model_name)
        print(self.model.summary())
        return self.model