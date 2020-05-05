# 21CNN
Training CNNs & RNNs with 21cmFAST images.

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
| Flag | Choices | Default | Description|
| ---:| :------:| :------:| :------ |
| `simple_run` | {0, 1} | 0 | If 1, runs hyperparameters defined in `py21cnn.hyperparamters.HP_simple`. Else, runs hyperparameters defined with `HyperparameterIndex`. |
| `dimensionality` | {2, 3} | 3 | Data dimensionality. |
| `removed_average` | {0, 1} | 1 | Is average for each redshift removed from the lightcone? Doesn't perform calculation, data loading flag. |
| `Zmax` | | 30 | Maximum redshift considered. Doesn't perform the actual cut, data loading flag. |
| `data_location` |  | `data/` | Database location. |
| `saving_location` |  | `models/` | Where to save models, summary, logs. |
| `logs_location` |  | `logs/` | Location to save Tensorboard logs. |
| `model` |  | `RNN.SummarySpace3D` | Module and class name of the desired model, defined in `py21cnn.architectures`. |
| `model_type` | | | RNN and CNN models require different "time dimension" location. In the case it's not specified, inherits module name of the `model` flag. |
| `HyperparameterIndex` | {0, ..., N-1} | 0 | Picks hyperparameters from the list defined in `py21cnn.hyperparameters.HP`. Value is the index of cartesian product of all hyperparameters, where N is the total number of combinations. |
| `epochs` | | 200 | Number of epochs to train in the particular run. |
| `max_epochs` | | | Total number of epochs to train, LR scheduler depends on it. Has to be larger than `epochs`. If not specified, is set to `epochs`. |
| `LR_correction` | {0, 1} | 1 | In the case of multi-gpu training, should learning rate be multiplied by number of GPUs? |
| `file_prefix` | | ` ` | File prefix to all outputs of the program - saved models, logs. |
| `warmup` | | 0 | In the case of multi-gpu training, for how many epochs to linearly increase learining rate? |
| `tf` | | 1 | Tensorflow version, not working properly with tf2 at the moment. |

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
| Flag | Choices | Default | Description|
| ---:| :------:| :------:| :------ |
| `N_walker` | | 10000 | Number of different realizations of the 21cmFAST, which are funnily defined by walker vairable. |
| `N_slice` | | 4 | Number of slices to train on. By default, from one simulations, 4 slices are created, we can take all or some of them. |
| `N_noise` | | 10 | Number of noise realizations to train on. |
| `X_fstring` | | <sup id="a1">[\*](#f1)</sup> | Combining above parameters into f-string that defines name of the files in database. |
| `X_shape` | | `25,25,526` | Shape of the data. |
| `Y_filename` | | `NormalizedParams` | Filename for _labels_. |
| `pTVT` | | `0.8,0.1,0.1` | Data dimensionality. |
| `dimensionality` | {2, 3} | 3 | Train, validation, test division. |
| | | | |
| `simple_run` | | | _-\|\|-_ |
| `data_location` | | | _-\|\|-_|
| `saving_location`| | | _-\|\|-_|
| `logs_location`| | | _-\|\|-_|
| `model`| | | _-\|\|-_|
| `model_type`| | | _-\|\|-_|
| `HyperparameterIndex`| | | _-\|\|-_|
| `epochs`| | | _-\|\|-_|
| `max_epochs`| | | _-\|\|-_|
| `gpus`| | | _-\|\|-_|
| `LR_correction`| | | _-\|\|-_|
| `file_prefix`| | | _-\|\|-_|
| `warmup`| | | _-\|\|-_|
| `tf`| | | _-\|\|-_|

<b id="f1">[\*](#a1)</b> `lightcone_depthMhz_0_walker_{:04d}_slice_{:d}_seed_{:d}`
