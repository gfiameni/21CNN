import numpy as np
import sys
import os

############################
### KERAS BACKEND        ###
############################
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tensorflow as tf
from keras import backend as K

K.set_image_data_format('channels_last')

# ############################
# ### LIMIT GPU USAGE      ###
# ############################
# #https://www.tensorflow.org/guide/gpu
# MaxGB = 10.5
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate MaxGB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(MaxGB*1024))])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

############################
### LOAD DADA: LIGHTCONE ###
############################
DataFilepath = "../../data/"
DataXname = "database5_float32.npy"
pTrain = 0.8
pDev = 0.1
pTest = 0.1
tophat = [2, 2]
Zmax = 12

trainX = np.load(DataFilepath + 'train/X_'+"0.8_tophat22_Z12_meanZ_database5_float32.npy")
trainY = np.load(DataFilepath + 'train/Y_'+"0.8_tophat22_Z12_meanZ_database5_float32.npy")
testX  = np.load(DataFilepath + 'test/X_'+"0.1_tophat22_Z12_meanZ_database5_float32.npy")
testY  = np.load(DataFilepath + 'test/Y_'+"0.1_tophat22_Z12_meanZ_database5_float32.npy")
devX   = np.load(DataFilepath + 'dev/X_'+"0.1_tophat22_Z12_meanZ_database5_float32.npy")
devY   = np.load(DataFilepath + 'dev/Y_'+"0.1_tophat22_Z12_meanZ_database5_float32.npy")
### adjustment of data dimention -> channels_last
trainX = trainX[..., np.newaxis]
testX = testX[..., np.newaxis]
devX = devX[..., np.newaxis]


######################
### CREATING MODEL ###
######################
from architectures import NGillet
model = NGillet.modelNN(input_shape = trainX.shape[1:])

######################
### LEARNING PHASE ###
######################
from keras.optimizers import RMSprop

### Network PARAMETERS
### LOSS FUNCTION
loss = 'mean_squared_error' ### classic loss function for regression, see also 'mae'
### DEFINE THE OPTIMIZER
# optimizer = 'RMSprop' #'adagrad'  #'adadelta' #'adam' # 'adamax' # 'Nadam' # 'RMSprop' # sgd
opt = RMSprop(lr=0.1)
### DEFINE THE LEARNING RATE
factor=0.5
patience=5

### set the learning rate callback
callbacks_list=[]
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
lrate = ReduceLROnPlateau( monitor='loss', factor=factor, patience=patience )
callbacks_list.append( lrate )

### to print the Learning Rate
from keras.callbacks import Callback, EarlyStopping
class LR_tracer(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        lr = K.eval( self.model.optimizer.lr )
        print( ' LR: %.10f '%(lr) )
callbacks_list.append( LR_tracer() )

### R2 coefficient
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

### STOP when it stop to learn
early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
callbacks_list.append( early_stopping )

### model compilations
model.compile( loss=loss,
               optimizer=opt,
               metrics=[coeff_determination] )

### THE LEARNING FUNCTION
batch_size = 2**3 ### number of sub sample, /!\ has to be a diviseur of the training set
epochs = 200   ### number of passage over the full data set

history = model.fit( trainX, trainY,
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=callbacks_list,
                     validation_data=(devX, devY),
                     verbose=True )



### save files
model_file = '2D_Filter55_1batchNorm'
history_file = model_file + '_history'
prediction_file = model_file + '_pred'
prediction_file_val = model_file + '_pred_val'

### save folder
CNN_folder = 'data_save/NGillet/'
np.save( CNN_folder + history_file, history.history )

########################
### SAVING THE MODEL ###
########################

def save_model( model, fileName ):
    """
    save a model
    """
    ### save the model
    model_json = model.to_json(  )
    with open( fileName+'.json', 'w' ) as json_file:
        json_file.write( model_json )
    ### save the weights
    model.save_weights( fileName+'_weights.h5' )

save_model( model, CNN_folder + model_file )

##################
### PREDICTION ###
##################

predictions = model.predict( testX, verbose=True )

### PRINT SCORE
# if all4:  
print( 'R2: ', 1 - (((predictions[:,0] - testY[:,0])**2).sum(axis=0)) / ((predictions[:,0] - predictions[:,0].mean(axis=0) )**2).sum(axis=0) )
print( 'R2: ', 1 - (((predictions[:,1] - testY[:,1])**2).sum(axis=0)) / ((predictions[:,1] - predictions[:,1].mean(axis=0) )**2).sum(axis=0) )
print( 'R2: ', 1 - (((predictions[:,2] - testY[:,2])**2).sum(axis=0)) / ((predictions[:,2] - predictions[:,2].mean(axis=0) )**2).sum(axis=0) )
print( 'R2: ', 1 - (((predictions[:,3] - testY[:,3])**2).sum(axis=0)) / ((predictions[:,3] - predictions[:,3].mean(axis=0) )**2).sum(axis=0) )
# else: 
#     print( 'R2: ', 1 - (((predictions[:,0] - Param_test[:,paramNum])**2).sum(axis=0)) / ((predictions[:,0] - predictions[:,0].mean(axis=0) )**2).sum(axis=0) )

np.save( CNN_folder + prediction_file, predictions )

### Predict the validation, to be use only at the end end end ....
predictions_val = model.predict( devX, verbose=True )
np.save( CNN_folder + prediction_file_val, predictions_val )