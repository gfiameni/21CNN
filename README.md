# 21CNN
Training CNNs & RNNs with 21cmFAST images

## Small database
### Example
```bash
python run.py \
        --removed_average 1 \ 
        --dimensionality 3 \  
        --data_location /path/to/data/ \
        --saving_location /path/to/models/ \
        --logs_location /path/to/logs/ \ 
        --model CNN.basic3D \ 
        --HyperparameterIndex 14 \ 
        --epochs 1000 \
```
### Options
```bash
'--simple_run', type=int, choices=[0, 1], default = 0
        If 1, runs hyperparameters defined in py21cnn.hyperparamters.HP_simple.
        Else, runs hyperparameters defined with --HyperparameterIndex.
'--dimensionality', type=int, choices=[2, 3], default=3
        data dimensionality
'--removed_average', type=int, choices=[0, 1], default=1
        Is average for each redshift removed from the lightcone?
        Doesn't perform calculation, data loading flag.
'--Zmax', type=int, default=30
        Maximum redshift considered.
        Doesn't perform the actual cut, data loading flag.
        
'--data_location', type=str, default="data/"
        Database location.
'--saving_location', type=str, default="models/"
        Where to save models, summary, logs.
'--logs_location', type=str, default="logs/"
        Location for tensorboard logs.

'--model', type=str, default="RNN.SummarySpace3D"
        Module and class name of the desired model, defined in py21cnn.architectures.
'--model_type', type=str
        RNN and CNN models require different "time dimension" location.
        In the case it's not specified, inherits module name of the '--model' flag.
'--HyperparameterIndex', type=int, default=0
        Picks hyperparameters from the list defined in py21cnn.hyperparameters.HP.
        Value is the index of cartesian product of all hyperparameters.
'--epochs', type=int, default=200
        Number of epochs to train in the particular run.
'--max_epochs', type=int
        Total number of epochs to train, LR scheduler depends on it. Has to be larger than '--epochs'.
        If not specified, is set to '--epochs'.
'--gpus', type=int, default=1
        Number of GPUs. 
'--LR_correction', type=int, choices=[0, 1], default=1
        In the case of multi-gpu training, should learning rate be multiplied by number of GPUs?
'--file_prefix', type=str, default=""
        File prefix to all outputs of the program - saved models, logs.
'--warmup', type=int, default=0
        In the case of multi-gpu training, for how many epochs to linearly increase learining rate?
'--tf', type=int, choices=[1, 2], default=1
        Tensorflow version, not working properly with tf2 at the moment
```

## Large database
### Example
For large database, where each lightcone is in a seperate `.npy` file:
```bash
 python large_run.py \
        --N_walker 100 \ 
        --N_slice 1 \
        --N_noise 2 \
        --model playground.SummarySpace3D_simple \
        --model_type RNN \
        --HyperparameterIndex 3 \
        --epochs 10 \
        --gpus 1 \
        --data_location /path/to/data/ \
        --saving_location /path/to/models/ \
        --logs_location /path/to/logs/ \
```
### Options
```bash
'--N_walker', type=int, default=10000
        Number of different realizations of the 21cmFAST, which are funnily defined by walker vairable.
'--N_slice', type=int, default=4
        Number of slices to train on.
        By default, from one simulations, 4 slices are created, we can take all or some of them.
'--N_noise', type=int, default=10
        Number of noise realizations to train on.
'--X_fstring', type=str, default = "lightcone_depthMhz_0_walker_{:04d}_slice_{:d}_seed_{:d}"
        Combining above parameters into f-string that defines name of the files in database.
'--X_shape', type=str, default="25,25,526"
        Shape of the data.
'--Y_filename', type=str, default = "NormalizedParams"
        Filename for "labels"
'--pTVT', type=str, default = "0.8,0.1,0.1"
        Train, validation, test division.
        
'--simple_run', type=int, choices=[0, 1], default = 0
'--data_location', type=str, default="data/"
'--saving_location', type=str, default="models/"
'--logs_location', type=str, default="logs/"
'--model', type=str, default="RNN.SummarySpace3D"
'--model_type', type=str, default=""
'--HyperparameterIndex', type=int, default=0
'--epochs', type=int, default=200
'--max_epochs', type=int
'--gpus', type=int, default=1
'--LR_correction', type=int, choices=[0, 1], default=1
'--file_prefix', type=str, default=""
'--warmup', type=int, default=0
'--tf', type = int, choices = [1, 2], default = 1
```
