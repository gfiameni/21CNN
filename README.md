# 21CNN
Training CNNs & RNNs with 21cmFAST images

## Run example
```bash
python run.py --removed_average 1 \ # data parameter, if average is removed for each redshift
              --dimensionality 3 \  # 3 for 3D models, 2 for 2D
              --data_location /path/to/data/ \
              --saving_location /path/to/models/ \
              --logs_location /path/to/logs/ \ # tensorboard logs
              --model CNN.basic3D \ # model name - has to be the same as the class name in src/py21cnn/architectures/
              --HyperparameterIndex 14 \ # all possible hyperparameters are enumerated with this index
              --epochs 1000 \ # number of epochs to train - if model exists in /path/to/models/, it runs for that many aditional epochs
#              --gpus 10 \ # number of gpus to use, default 1
#              --multi_gpu_correction 2 \ # see code
#              --patience 10 \ # number of epochs for keras.callbacks.ReduceLROnPlateau, default 10
```
