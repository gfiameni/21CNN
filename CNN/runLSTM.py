import numpy as np
import sys
import os
import json
import tensorflow as tf
import keras

os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf

#setting up GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1. #setting the percentage of GPU usage
config.gpu_options.visible_device_list = "0" #for picking only some devices
config.gpu_options.allow_growth = True

#passing tf session to keras!
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=config)) 

from keras import backend as K
K.set_image_data_format('channels_last')


#working with sliced data shape (10000, 4, 25, 25, ~500)
DataLoc = "../../data/"
XName = "data3D_boxcar444_sliced22_float32.npy"
YName = "databaseParams_float32.npy"
YbackupName = "databaseParams_min_max.txt"

X = {}
Y = {}
for sort in ["train", "val", "test"]:
    X[sort] = np.load(f"{DataLoc}{sort}/X_{XName}")
    X[sort] = X[sort][..., np.newaxis]
    
for sort in ["train", "val", "test"]:
    Y[sort] = np.load(f"{DataLoc}{sort}/Y_{XName}")
    shape = Y[sort].shape
    Y[sort] = Y[sort][..., np.newaxis]
    Y[sort] = np.broadcast_to(Y[sort], shape + (X[sort].shape[1],))
    Y[sort] = np.swapaxes(Y[sort], -1, -2)
    print(Y[sort].shape)
    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense
from tensorflow.keras.layers import Conv3D, Conv2D, Conv1D, GlobalAveragePooling2D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization, Flatten

class HyperParams1:
    def __init__(
        self,
        LossDecay = 0.9,
        LossLength = 10,
#         SummaryDimension = 16,
        ):
        self.LossDecay = LossDecay
        self.LossLength = LossLength
#         self.SummaryDimension = SummaryDimension

HP = HyperParams1()
# input_shape = X["train"].shape[1:]
input_shape = (526, 25, 25, 1)

def Loss1(y_true, y_pred):
    length = input_shape[0]
    weights = HP.LossDecay**np.arange(HP.LossLength - 1, -1, -1)
    weights = np.append(np.zeros(length-HP.LossLength), weights)
    weights = np.broadcast_to(weights[..., np.newaxis], (length, 4))
#     weights = tf.math.pow(tf.constant([HP.LossDecay], dtype=tf.float32), tf.range(tf.shape(y_true)[-2] - 1, -1, -1, dtype=tf.float32))
#     weights = HP.LossDecay**np.arange(np.array(y_true).shape[-2] - 1, -1, -1)
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    sq = weights * tf.keras.backend.square(y_pred - y_true)
    return tf.keras.backend.mean(sq, axis=-1)


model1_simple = Sequential()
model1_simple.add(TimeDistributed(Conv2D(filters=32, kernel_size=(7, 7), activation='relu'), input_shape = input_shape, name="Conv0"))
model1_simple.add(TimeDistributed(GlobalAveragePooling2D(), name="GlobalPool"))
model1_simple.add(TimeDistributed(Flatten(), name="Flatten"))
#RNN part
model1_simple.add(LSTM(32, return_sequences=True, name="DenseLSTM0"))
model1_simple.add(LSTM(16, return_sequences=True, name="DenseLSTM1"))
# model1_simple.add(LSTM(16, return_sequences=True, name="DenseLSTM2"))
model1_simple.add(TimeDistributed(Dense(4), name="out"))
# model1.add(Dense(4))
model1_simple.summary()

model1_simple.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1), 
               loss=Loss1)