import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Dense, Conv2D
from tensorflow.keras.models import Sequential

model = Sequential(Dense(1, input_shape=(10,)))

import numpy as np

X = np.random.rand(100, 10)
Y = np.random.rand(100)
Xv = np.random.rand(10, 10)
Yv = np.random.rand(10)

def Loss1(y_true, y_pred):
    sq = 2 * tf.keras.backend.square(y_pred - y_true)
    return tf.keras.backend.mean(sq, axis=-1)


learning_rate = 0.1
optimizer = keras.optimizers.RMSprop(lr = learning_rate)
batch_size = 20
epochs = 100


def coeff_determination(y_true, y_pred):
    SS_res =  keras.backend.sum(keras.backend.square( y_true-y_pred )) 
    SS_tot = keras.backend.sum(keras.backend.square( y_true - keras.backend.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + keras.backend.epsilon()) )

model.compile( loss=Loss1,
               optimizer=optimizer,
               metrics = [coeff_determination])

model.summary()

class LR_tracer(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        lr = keras.backend.eval( self.model.optimizer.lr )
        print( ' LR: %.10f '%(lr) )

callbacks = [
    LR_tracer(),
    keras.callbacks.ModelCheckpoint("model_best.hdf5", monitor='val_loss', save_best_only=True, verbose=True),
    keras.callbacks.ModelCheckpoint("model_last.hdf5", monitor='val_loss', save_best_only=False, verbose=True), 
    keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.5, patience=10, verbose=True),
    keras.callbacks.CSVLogger('test4.log', separator=',', append=True),
]

history = model.fit( X, Y,
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=callbacks,
                     validation_data=(Xv, Yv),
                     verbose=2 )