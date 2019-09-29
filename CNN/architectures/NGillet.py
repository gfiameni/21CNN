import numpy as np
from time import time
import sys, argparse, textwrap

from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU

from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential


# ### save files
# model_template = '%s_2D'
# model_file = model_template%(paramName[paramNum]) + name
# history_file = model_file + '_history'
# prediction_file = model_file + '_pred'
# prediction_file_val = model_file + '_pred_val'

# ### save folder
# CNN_folder = 'CNN_save/'

### to use openMP
### export OMP_NUM_THREADS=2

####################################
### CONVOLUTIONAL NEURAL NETWORK ###
####################################    
def modelNN(
    input_shape,
    padding = 'valid', ### 'same' or 'valid
    filter_size = (5, 5),
    pool_size = (2,2),
    activation = 'relu', ### 'linear' 'relu'
    LeackyRelu_alpha = 0,
    use_bias = True,
    batchNorm = False,
    FirstbatchNorm = True,
    use_dropout = 0.2,
    Nfilter1 = 8, ### First convolution
    Nfilter2 = 16, ### 2nd convolution
    Nfilter3 = 64, ### First Dense
    batch_size = 20, ### number of sub sample, /!\ has to be a diviseur of the training set
    epochs = 200   ### number of passage over the full data set
    ):
    if( batchNorm ):
        use_bias=False
        activation = 'linear' ### 'linear' 'relu'
        
    if( LeackyRelu_alpha ):
        activation = 'linear' ### 'linear' 'relu'

    model = Sequential()

    ### CONV 1
    model.add( Convolution2D( Nfilter1, filter_size, activation=activation, 
                            input_shape=input_shape, name='Conv-1', padding=padding, use_bias=use_bias ) )
    if( batchNorm ):
        model.add( BatchNormalization() )
    if( LeackyRelu_alpha ):
        model.add( LeakyReLU(alpha=LeackyRelu_alpha) )
    if( ( batchNorm ) and not(LeackyRelu_alpha) ):
        model.add( Activation('relu') )
            
    ### MAXPOOL 1
    model.add( MaxPooling2D( pool_size=pool_size, name='Pool-1' ) )

    ### CONV 2
    model.add( Convolution2D( Nfilter2, filter_size, activation=activation, 
                            name='Conv-2', padding=padding, use_bias=use_bias ) )
    if( batchNorm ):
        model.add( BatchNormalization() )
    if( LeackyRelu_alpha ):
        model.add( LeakyReLU(alpha=LeackyRelu_alpha) )
    if( batchNorm and not(LeackyRelu_alpha) ):
        model.add( Activation('relu') )
            
    ### MAXPOOL 2
    model.add( MaxPooling2D( pool_size=pool_size, name='Pool-2' ) )

    ### FLATTEN
    model.add( Flatten( name='Flat' ) )
    if use_dropout: 
        model.add( Dropout(use_dropout) )
            
    ### DENSE 1
    model.add( Dense( Nfilter3, activation=activation, name='Dense-1', use_bias=use_bias ) )
    if( batchNorm or FirstbatchNorm ):
        model.add( BatchNormalization() )
    if( ( batchNorm or FirstbatchNorm ) and not(LeackyRelu_alpha) ):
        model.add( Activation('relu') )
        
    ### DENSE 2
    model.add( Dense( Nfilter2, activation=activation, name='Dense-2', use_bias=use_bias ) )
    if( batchNorm ):
        model.add( BatchNormalization() )
    if( batchNorm and not(LeackyRelu_alpha) ):
        model.add( Activation('relu') )
        
    ### DENSE 3
    model.add( Dense( Nfilter1, activation=activation, name='Dense-3', use_bias=use_bias ) )
    if( batchNorm ):
        model.add( BatchNormalization() )
    if( batchNorm and not(LeackyRelu_alpha) ):
        model.add( Activation('relu') )
            
    ### DENSE OUT
    model.add( Dense( 4, activation='linear', name='Out' ) )
        
    ##############################    
    model.summary(line_length=120) 
