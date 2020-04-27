# 21CNN
Training CNNs & RNNs with 21cmFAST images

## Run example
For small database run the following:
```bash
python run.py \
        --removed_average 1 \ # data parameter, if average is removed for each redshift
        --dimensionality 3 \  # 3 for 3D models, 2 for 2D
        --data_location /path/to/data/ \
        --saving_location /path/to/models/ \
        --logs_location /path/to/logs/ \ # tensorboard logs
        --model CNN.basic3D \ # model name - has to be the same as the class name in src/py21cnn/architectures/
        --HyperparameterIndex 14 \ # all possible hyperparameters are enumerated with this index
        --epochs 1000 \ # number of epochs to train - if model exists in /path/to/models/, it runs for that many aditional epochs
#        --gpus 10 \ # number of gpus to use, default 1
#        --multi_gpu_correction 2 \ # see code
#        --patience 10 \ # number of epochs for keras.callbacks.ReduceLROnPlateau, default 10
```

For large database, where each lightcone is in a seperate `.npy` file:
```bash
 python large_database_run.py \
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
        --logs_location /path/to/logs/ \ # tensorboard logs
```
