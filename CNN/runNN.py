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

############################
### LOAD DADA: LIGHTCONE ###
############################
DataFilepath = "../../data/"
DataXname = f"database5_float32.npy"
pTrain = 0.8
pDev = 0.1
pTest = 0.1
tophat = [2, 2]
Zmax = 12

trainX = np.load(f"{DataFilepath}train/X_{pTrain:.1f}_tophat{tophat[0]}{tophat[1]}_Z{Zmax}_meanZ_{DataXname}")
trainY = np.load(f"{DataFilepath}train/Y_{pTrain:.1f}_tophat{tophat[0]}{tophat[1]}_Z{Zmax}_meanZ_{DataXname}")
testX  = np.load(f"{DataFilepath}test/X_{pTest:.1f}_tophat{tophat[0]}{tophat[1]}_Z{Zmax}_meanZ_{DataXname}")
testY  = np.load(f"{DataFilepath}test/Y_{pTest:.1f}_tophat{tophat[0]}{tophat[1]}_Z{Zmax}_meanZ_{DataXname}")
devX   = np.load(f"{DataFilepath}dev/X_{pTest:.1f}_tophat{tophat[0]}{tophat[1]}_Z{Zmax}_meanZ_{DataXname}")
devY   = np.load(f"{DataFilepath}dev/Y_{pTest:.1f}_tophat{tophat[0]}{tophat[1]}_Z{Zmax}_meanZ_{DataXname}")
### adjustment of data dimention -> channels_last
trainX = trainX[..., np.newaxis]
testX = testX[..., np.newaxis]
devX = devX[..., np.newaxis]


######################
### LEARNING PHASE ###
######################

### DEFINE THE LEARNING RATE

### set the learning rate callback
callbacks_list=[]
if( 1 ):
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
               optimizer=optimizer,
               metrics=[coeff_determination] )

### THE LEARNING FUNCTION
history = model.fit( LC_train, Param_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=callbacks_list,
                     validation_data=( LC_test, Param_test ),
                     verbose=True )

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

predictions = model.predict( LC_test, verbose=True )

### PRINT SCORE
if all4:  
    print( 'R2: ', 1 - (((predictions[:,0] - Param_test[:,0])**2).sum(axis=0)) / ((predictions[:,0] - predictions[:,0].mean(axis=0) )**2).sum(axis=0) )
    print( 'R2: ', 1 - (((predictions[:,1] - Param_test[:,1])**2).sum(axis=0)) / ((predictions[:,1] - predictions[:,1].mean(axis=0) )**2).sum(axis=0) )
    print( 'R2: ', 1 - (((predictions[:,2] - Param_test[:,2])**2).sum(axis=0)) / ((predictions[:,2] - predictions[:,2].mean(axis=0) )**2).sum(axis=0) )
    print( 'R2: ', 1 - (((predictions[:,3] - Param_test[:,3])**2).sum(axis=0)) / ((predictions[:,3] - predictions[:,3].mean(axis=0) )**2).sum(axis=0) )
else: 
    print( 'R2: ', 1 - (((predictions[:,0] - Param_test[:,paramNum])**2).sum(axis=0)) / ((predictions[:,0] - predictions[:,0].mean(axis=0) )**2).sum(axis=0) )

np.save( CNN_folder + prediction_file, predictions )

### Predict the validation, to be use only at the end end end ....
predictions_val = model.predict( LC_val, verbose=True )
np.save( CNN_folder + prediction_file_val, predictions_val )