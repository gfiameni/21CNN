import numpy as np
import sys
import os

############################
### KERAS BACKEND and LIMIT GPU
############################
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf

#setting up GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85 #setting the percentage of GPU usage
#config.gpu_options.visible_device_list = "0" #for picking only some devices

#passing tf session to keras!
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=config)) 

from keras import backend as K
K.set_image_data_format('channels_last')

######### not working for now
# ############################
# ### LIMIT GPU USAGE      ###
# ############################
# #https://www.tensorflow.org/guide/gpu
# MaxGB = 10
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
RemovedMean = False
if RemovedMean:
    rm = "meanZ_"
else:
    rm = ""
pTrain = 0.8
pDev = 0.1
pTest = 0.1
tophat = [2, 2]
Zmax = 30
DataType = ("tophat{}{}_Z{}_{}" + DataXname).format(*tophat, Zmax, rm)
print(DataType)
TDTfile = "{}/{}_{:.1f}_" + DataType
trainX = np.load(DataFilepath + TDTfile.format("train", "X", pTrain))
trainY = np.load(DataFilepath + TDTfile.format("train", "Y", pTrain))
testX  = np.load(DataFilepath + TDTfile.format("test", "X", pTest))
testY  = np.load(DataFilepath + TDTfile.format("test", "Y", pTest))
devX   = np.load(DataFilepath + TDTfile.format("dev", "X", pDev))
devY   = np.load(DataFilepath + TDTfile.format("dev", "Y", pDev))
### adjustment of data dimension -> channels_last
trainX = trainX[..., np.newaxis]
testX = testX[..., np.newaxis]
devX = devX[..., np.newaxis]


######################
### CREATING MODEL ###
######################
from architectures import NGillet
# model = NGillet.modelNN(input_shape = trainX.shape[1:], 
#                         # filter_size=(5, 5), 
#                         # Nfilter1=16, Nfilter2=32, Nfilter3=64, 
#                         # FirstbatchNorm=False,
#                         # use_dropout=0,
#                         )
model = NGillet.modelNN_deeper(input_shape = trainX.shape[1:], 
                        # filter_size=(5, 5), 
                        Nfilter1=16, Nfilter2=32, Nfilter3=64, 
                        # FirstbatchNorm=False,
                        # use_dropout=0,
                        )
#load some old weights
tophat_old = [2, 2]
Zmax_old = 30
rm_old = "meanZ_"
model_file = "tophat22_Z30_meanZ_deeper2D_Filter55_1batchNorm"
weights_file = model_file + '_weights.h5'
CNN_folder = "data_save/NGillet/tophat{}{}_Z{}_{}deeper_good/".format(*tophat_old, Zmax_old, rm_old)
model.load_weights(CNN_folder + weights_file)

######################
### LEARNING PHASE ###
######################
from keras.optimizers import RMSprop

### Network PARAMETERS
### LOSS FUNCTION
loss = 'mean_squared_error' ### classic loss function for regression, see also 'mae'
### DEFINE THE OPTIMIZER
# optimizer = 'RMSprop' #'adagrad'  #'adadelta' #'adam' # 'adamax' # 'Nadam' # 'RMSprop' # sgd
opt = RMSprop(lr=0.01)
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
batch_size = 20 ### number of sub sample, /!\ has to be a diviseur of the training set
epochs = 200   ### number of passage over the full data set

history = model.fit( trainX, trainY,
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=callbacks_list,
                     validation_data=(devX, devY),
                     verbose=True )



### save files
model_file = "2D_Filter55_1batchNorm"
history_file = model_file + '_history'
prediction_file = model_file + '_pred'
prediction_file_val = model_file + '_pred_val'

### save folder
CNN_folder = "data_save/NGillet/tophat{}{}_Z{}_{}deeper/".format(*tophat, Zmax, rm)
os.makedirs(CNN_folder, exist_ok=True)
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