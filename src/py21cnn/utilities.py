from . import ctx

# from . import formatting
# Filters = formatting.Filters
from .formatting import Filters
import os
import time
import copy
import hashlib
import numpy as np
import io
import tensorflow as tf
# import keras
from tensorflow import keras
# from tensorboard.plugins.hparams import api as hp
import horovod.tensorflow.keras as hvd
# sess = keras.backend.get_session()

class Data:
    def __init__(
        self,
        filepath = None,
        dimensionality = 2,
        removed_average = True,
        normalized = True,
        Zmax = 30,
        filetype = 'float32',
        formatting = [],
        X = {},
        Y = {},
        ):
        self.filepath = filepath
        self.dimensionality = dimensionality
        self.removed_average = removed_average
        self.normalized = normalized
        self.Zmax = Zmax
        self.filetype = filetype
        self.X = X
        self.Y = Y
        if len(formatting) == 0:
            default_formatting = ['clipped_-250_+50', 'NaN_removed', 'TVT_parameterwise']
            if self.dimensionality == 2:
                default_formatting.append('boxcar44')
                default_formatting.append('10slices')
            if self.dimensionality == 3:
                default_formatting.append('boxcar444')
                default_formatting.append('sliced22')
            # default_formatting.sort()
            self.formatting = default_formatting
        else:
            # formatting.sort()
            self.formatting = formatting

    def __str__(self):
        self.formatting.sort()
        S = f"dim:{self.dimensionality}__removed_average:{self.removed_average}__normalized:{self.normalized}__Zmax:{self.Zmax}__dtype:{self.filetype}"
        for i in self.formatting:
            S += f"__{i}"
        return S
    
    def hash(self):
        return hashlib.md5(self.__str__().encode()).hexdigest()

    def load(self, model_type = None, pTVT = [0.8, 0.1, 0.1]):
        Hash = self.hash()
        for p, key in zip(pTVT, ['train', 'val', 'test']):
            self.X[key] = np.load(f"{self.filepath}X_{key}_{p:.2f}_{Hash}.npy")
            if model_type == "RNN":
                self.X[key] = np.swapaxes(self.X[key], -1, -self.dimensionality) #swapping to get time dimension on right place
            self.X[key] = self.X[key][..., np.newaxis]
            self.Y[key] = np.load(f"{self.filepath}Y_{key}_{p:.2f}_{Hash}.npy")
        self.shape = self.X['test'].shape[1:]
        self.TrainExamples = self.X['test'].shape[0]
        # return self.X, self.Y
        self.steps_per_epoch = self.X["train"].shape[0] // ctx.inputs.gpus // ctx.HP.BatchSize
        self.validation_steps = self.X["val"].shape[0] // ctx.HP.BatchSize

    def saveTVT(self, pTVT = [0.8, 0.1, 0.1]):
        Hash = self.hash()
        if len(self.X.keys()) == 0 or len(self.Y.keys()) == 0:
            raise ValueError('X or Y are empty dicts: not saving that')
        for p, key in zip(pTVT, ['train', 'val', 'test']):
            np.save(f"{self.filepath}X_{key}_{p:.2f}_{Hash}.npy", self.X[key])
            np.save(f"{self.filepath}Y_{key}_{p:.2f}_{Hash}.npy", self.Y[key])


class LargeData:
    def __init__(
        self,
        dimensionality = 2,
        removed_average = True,
        normalized = True,
        Zmax = 30,
        filetype = 'float32',
        formatting = [],
        noise = ["tools21cm", "SKA1000"],
        shape = None,
        load_all = False,
        ):
        self.dimensionality = dimensionality
        self.removed_average = removed_average
        self.normalized = normalized
        self.Zmax = Zmax
        self.filetype = filetype
        if len(formatting) == 0:
            default_formatting = ['clipped_-250_+50', 'NaN_removed', 'TVT_parameterwise']
            if self.dimensionality == 2:
                default_formatting.append('boxcar44')
                default_formatting.append('10slices')
            if self.dimensionality == 3:
                default_formatting.append('boxcar444')
                default_formatting.append('sliced22')
            # default_formatting.sort()
            self.formatting = default_formatting
        else:
            # formatting.sort()
            self.formatting = formatting
        self.noise = noise + [f"walkers_{ctx.inputs.N_walker}", f"slices_{ctx.inputs.N_slice}", f"noise_{ctx.inputs.N_noise}"]
        self.shape = shape
        self.load_all = load_all

    def __str__(self):
        self.formatting.sort()
        self.noise.sort()
        S = f"dim:{self.dimensionality}__removed_average:{self.removed_average}__normalized:{self.normalized}__Zmax:{self.Zmax}__dtype:{self.filetype}"
        for i in self.formatting:
            S += f"__{i}"
        for i in self.noise:
            S += f"__{i}"
        return S
    
    def hash(self):
        return hashlib.md5(self.__str__().encode()).hexdigest()

    def load(self):
        permutation = Filters.constructIndexArray(ctx.inputs.N_walker, *ctx.inputs.pTVT, 1312)
        # print(permutation)
        Y = np.load(f"{ctx.inputs.data_location}{ctx.inputs.Y_filename}.npy")
        self.y_max = np.amax(Y, axis = 0)
        self.y_min = np.amin(Y, axis = 0)
        ctx.inputs.Y_shape = Y.shape[-1]
        self.partition = {
            "train": [], 
            "validation": [], 
            "test": []}
        self.noise_rolling_partition = {
            "train": [], 
            "validation": [], 
            "test": []}
        keys = list(self.partition.keys())
        for key in keys:
            for seed in range(ctx.inputs.N_noise):
                self.noise_rolling_partition[key].append([])
        self.labels = {}
        self.inputs = {}
        for walker in range(ctx.inputs.N_walker):
            for s in range(ctx.inputs.N_slice):
                for seed in range(ctx.inputs.N_noise):
                    ID = ctx.inputs.X_fstring.format(walker, s, seed)
                    self.partition[keys[permutation[walker]]].append(ID)
                    self.noise_rolling_partition[keys[permutation[walker]]][seed].append(ID)
                    self.labels[ID] = Y[walker]
                    if self.load_all == True:
                        self.inputs[ID] = np.load(f"{ctx.inputs.data_location}{ID}.npy")
        # print(self.partition)

        if ctx.inputs.noise_rolling == True:
            self.steps_per_epoch = len(self.noise_rolling_partition["train"][0]) // ctx.inputs.gpus // ctx.HP.BatchSize
            self.validation_steps = len(self.noise_rolling_partition["validation"][0]) // ctx.HP.BatchSize   
        else:         
            self.steps_per_epoch = len(self.partition["train"]) // ctx.inputs.gpus // ctx.HP.BatchSize
            self.validation_steps = len(self.partition["validation"]) // ctx.HP.BatchSize


class Data_tfrecord:
    def __init__(
        self,
        dimensionality = 2,
        removed_average = True,
        normalized = True,
        Zmax = 30,
        filetype = 'float32',
        formatting = [],
        noise = ["tools21cm", "SKA1000"],
        shape = None,
        load_all = False,
        ):
        self.dimensionality = dimensionality
        self.removed_average = removed_average
        self.normalized = normalized
        self.Zmax = Zmax
        self.filetype = filetype
        if len(formatting) == 0:
            default_formatting = ['clipped_-250_+50', 'NaN_removed', 'TVT_parameterwise']
            if self.dimensionality == 2:
                default_formatting.append('boxcar44')
                default_formatting.append('10slices')
            if self.dimensionality == 3:
                default_formatting.append('boxcar444')
                default_formatting.append('sliced22')
            # default_formatting.sort()
            self.formatting = default_formatting
        else:
            # formatting.sort()
            self.formatting = formatting
        self.noise = noise + [f"walkers_{ctx.inputs.N_walker}", f"slices_{ctx.inputs.N_slice}", f"noise_{ctx.inputs.N_noise}"]
        self.shape = shape
        self.load_all = load_all

    def __str__(self):
        self.formatting.sort()
        self.noise.sort()
        S = f"dim:{self.dimensionality}__removed_average:{self.removed_average}__normalized:{self.normalized}__Zmax:{self.Zmax}__dtype:{self.filetype}"
        for i in self.formatting:
            S += f"__{i}"
        for i in self.noise:
            S += f"__{i}"
        return S
    
    def hash(self):
        return hashlib.md5(self.__str__().encode()).hexdigest()

    def decode(self, serialized_example, model_type):
        """Parses an image and label from the given `serialized_example`."""
        print(serialized_example)
        example = tf.io.parse_single_example(
            serialized_example,
            features={
                'Xx': tf.io.FixedLenFeature([], tf.int64),
                'Xy': tf.io.FixedLenFeature([], tf.int64),
                'Xz': tf.io.FixedLenFeature([], tf.int64),
                'X': tf.io.FixedLenFeature([], tf.string),
                'Yx': tf.io.FixedLenFeature([], tf.int64),
                'Y': tf.io.FixedLenFeature([], tf.string),
            })
        xx = tf.cast(example['Xx'], tf.int64)
        xy = tf.cast(example['Xy'], tf.int64)
        xz = tf.cast(example['Xz'], tf.int64)
        x = tf.io.decode_raw(example['X'], tf.float32)
        x = tf.reshape(x, (xx, xy, xz))
        if model_type == "RNN":
            x = tf.transpose(x)
        x = tf.expand_dims(x, -1)
        yx = tf.cast(example['Yx'], tf.int64)
        y = tf.io.decode_raw(example['Y'], tf.float32)
        y = tf.reshape(y, (yx,))
        return x, y

    def create_shards(self, filenames, ds_type):
        shards = tf.data.Dataset.from_tensor_slices(filenames[0])
        if ds_type == "train":
            shards = shards.shuffle(len(filenames[0]))
        for i in range(1, len(filenames)):
            t_shards = tf.data.Dataset.from_tensor_slices(filenames[i])
            if ds_type == "train":
                t_shards = t_shards.shuffle(len(filenames[i]))
            shards = shards.concatenate(t_shards)
        #in the case of test database repeat only once, else repeat indefinitely
        if ds_type == "test":
            shards = shards.repeat(1)
        else:
            shards = shards.repeat()
        return shards

    def get_dataset(self, ds_type, filenames, model_type, batch_size, buffer_size, workers):
        """Read TFRecords files and turn them into a TFRecordDataset."""
        shards = self.create_shards(filenames, ds_type)
        if ds_type == "train":
            dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length = 5, block_length = 1)
        else:
            dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length = 1, block_length = 1)
        #   dataset = dataset.shuffle(buffer_size=8192)
        dataset = dataset.map(map_func = lambda x: self.decode(x, model_type), num_parallel_calls = workers).batch(batch_size = batch_size)
        dataset = dataset.prefetch(buffer_size = buffer_size)
        return dataset

    def load(self):
        shardsTVT = {
            "train": int(ctx.inputs.N_walker * ctx.inputs.N_slice * ctx.inputs.pTVT[0]) // 100,
            "validation": int(ctx.inputs.N_walker * ctx.inputs.N_slice * ctx.inputs.pTVT[1]) // 100,
            "test": int(ctx.inputs.N_walker * ctx.inputs.N_slice * ctx.inputs.pTVT[1]) // 100,
            }
        self.filenames = {
            "train": [],
            "validation": [],
            "test": []
            }
        for key in self.filenames.keys():
            for seed in range(ctx.inputs.N_noise):
                self.filenames[key].append([f"{ctx.inputs.data_location}{ctx.inputs.X_fstring.format(key, seed, i, shardsTVT[key]-1)}" for i in range(shardsTVT[key])])

        self.train_ds = self.get_dataset(
            "train",
            self.filenames["train"], 
            ctx.inputs.model_type, 
            batch_size = ctx.HP.BatchSize, 
            buffer_size = 16, 
            workers = ctx.inputs.workers)
        self.validation_ds = self.get_dataset(
            "validation",
            [self.filenames["validation"][0]], 
            ctx.inputs.model_type,
            batch_size = ctx.HP.BatchSize, 
            buffer_size = 16, 
            workers = ctx.inputs.workers)
        self.validation_ds = self.get_dataset(
            "test",
            self.filenames["train"], 
            ctx.inputs.model_type,
            batch_size = ctx.HP.BatchSize, 
            buffer_size = 16, 
            workers = ctx.inputs.workers)
        self.steps_per_epoch = len(self.filenames["train"]) * 100 // ctx.HP.BatchSize
        self.validation_steps = len(self.filenames["validation"]) * 100 // ctx.HP.BatchSize

class AuxiliaryHyperparameters:
    def __init__(
        self,
        model_name,
        Epochs = 200,
        MaxEpochs = 200,
        NoiseRolling = None,
        # Loss = {"instance": keras.losses.mean_squared_error, "name": "mse"},
        Loss = [keras.losses.mean_squared_error, "mse"],
        # Optimizer = {"instance": keras.optimizers.RMSprop(), "name": "RMSprop"},
        Optimizer = [keras.optimizers.RMSprop, "RMSprop", {}],
        LearningRate = 0.01,
        # ActivationFunction = {"instance": keras.activations.relu(), "name": "relu"},
        ActivationFunction = ["relu", {"activation": keras.activations.relu, "kernel_initializer": keras.initializers.he_uniform()}],
        BatchNormalization = True,
        Dropout = 0.2,
        ReducingLR = False, 
        BatchSize = 20,
        ):
        self.model_name = model_name
        self.Loss = Loss
        self.Optimizer = Optimizer
        self.LearningRate = LearningRate
        self.Optimizer[2]["lr"] = self.LearningRate
        self.ActivationFunction = ActivationFunction
        self.BatchNormalization = BatchNormalization
        self.Dropout = Dropout
        self.ReducingLR = ReducingLR
        self.BatchSize = BatchSize
        self.Epochs = Epochs
        self.MaxEpochs = MaxEpochs
        self.TensorBoard = {
            "Model": self.model_name,
            "LearningRate": self.LearningRate,
            "Dropout": self.Dropout,
            "BatchSize": self.BatchSize,
            "BatchNormalization": self.BatchNormalization,
            "Optimizer": self.Optimizer[1],
            "ActivationFunction": self.ActivationFunction[0],
            }
        self.NoiseRolling = NoiseRolling

    def __str__(self):
        S = f"Loss:{self.Loss[1]}__Optimizer:{self.Optimizer[1]}__LR:{self.LearningRate:.10f}__Activation:{self.ActivationFunction[0]}"
        S += f"__BN:{self.BatchNormalization}__dropout:{self.Dropout:.2f}__reduceLR:{self.ReducingLR}__Batch:{self.BatchSize:05d}__Epochs:{self.MaxEpochs:05d}"
        if self.NoiseRolling != None:
            S += f"__NoiseRolling:{self.NoiseRolling}"
        return S
    def hashstring(self):
        #differences from __str__ in not including epochs
        S = f"Loss:{self.Loss[1]}__Optimizer:{self.Optimizer[1]}__LR:{self.LearningRate:.10f}__Activation:{self.ActivationFunction[0]}"
        S += f"__BN:{self.BatchNormalization}__dropout:{self.Dropout:.2f}__reduceLR:{self.ReducingLR}__Batch:{self.BatchSize:05d}"
        if self.NoiseRolling != None:
            S += f"__NoiseRolling:{self.NoiseRolling}"        
        return S
    def hash(self):
        return hashlib.md5(self.hashstring().encode()).hexdigest()


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                list_IDs,
                labels,
                inputs,
                dimX, 
                dimY,
                data_filepath,
                model_type,
                batch_size,
                load_all,
                initial_epoch,
                N_noise,
                noise_rolling,
                iterations = None,
                n_channels=1,
                shuffle=True,
                ):
        self.model_type = model_type
        if self.model_type == "RNN":
            self.dimX = dimX[::-1]
        else:
            self.dimX = dimX
        self.dimY = dimY
        self.data_filepath = data_filepath
        self.batch_size = batch_size
        self.load_all = load_all
        self.labels = labels
        self.list_IDs = list_IDs
        self.inputs = inputs
        self.N_noise = N_noise
        self.initial_epoch = initial_epoch
        self.noise_index = self.initial_epoch % self.N_noise - 1
        self.noise_rolling = noise_rolling
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.iterations = iterations
        self.__len__()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.iterations == None:
            if self.noise_rolling == True:
                self.iterations = len(self.list_IDs[0]) // self.batch_size
            else:
                self.iterations =  len(self.list_IDs) // self.batch_size
        return self.iterations

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        if self.noise_rolling == True:
            list_IDs_temp = [self.list_IDs[self.noise_index][k] for k in indexes]
        else:
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # print(list_IDs_temp)

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        # return tf.data.Dataset.from_tensor_slices((X, y))
        return X, y

    def on_epoch_end(self):
        self.noise_index  = (self.noise_index + 1) % self.N_noise
        print("NOISE INDEX", self.noise_index)
        'Updates indexes after each epoch'
        if self.noise_rolling == True:
            # self.noise_index = (self.noise_index + 1) % self.N_noise
            self.indexes = np.arange(len(self.list_IDs[0]))
        else:
            self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def loadX(self, ID):
        if self.load_all == True:
            if self.model_type == "RNN":
                return np.swapaxes(self.inputs[ID], 0, -1)[..., np.newaxis]
            else:
                return self.inputs[ID][..., np.newaxis]
        else:
            if self.model_type == "RNN":
                return np.swapaxes(np.load(f"{self.data_filepath}{ID}.npy"), 0, -1)[..., np.newaxis]
            else:
                return np.load(f"{self.data_filepath}{ID}.npy")[..., np.newaxis]

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dimX, self.n_channels))
        y = np.empty((self.batch_size, self.dimY))

        for i, ID in enumerate(list_IDs_temp):           
            X[i,] = self.loadX(ID)
            y[i] = self.labels[ID]

        return X, y

class SimpleDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                list_IDs,
                labels,
                inputs,
                dimX, 
                dimY,
                data_filepath,
                model_type,
                batch_size,
                load_all,
                iterations = None,
                n_channels=1,
                filename = None,
                ):
        self.model_type = model_type
        if self.model_type == "RNN":
            self.dimX = dimX[::-1]
        else:
            self.dimX = dimX
        self.dimY = dimY
        self.data_filepath = data_filepath
        self.batch_size = batch_size
        self.load_all = load_all
        self.labels = labels
        self.list_IDs = list_IDs
        self.inputs = inputs
        self.n_channels = n_channels
        self.iterations = iterations
        # self.data = []
        if isinstance(filename, str):
            self.file = open(filename, "w")
        else:
            self.file = None
        self.__len__()
        self.indexes = np.arange(len(self.list_IDs))

    def __len__(self):
        'Denotes the number of batches'
        if self.iterations == None:
            self.iterations =  len(self.list_IDs) // self.batch_size
        return self.iterations

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        self.list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(self.list_IDs_temp)

        # print("IN SIMPLE DATA GENERATOR", self.list_IDs_temp)

        if self.file != None:
            for ID, label in zip(self.list_IDs_temp, y):
                self.file.write("{} {:.15f} {:.15f} {:.15f} {:.15f}\n".format(ID, *label))
            self.file.flush()
            os.fsync(self.file.fileno())

        return X, y

    def loadX(self, ID):
        if self.load_all == True:
            if self.model_type == "RNN":
                return np.swapaxes(self.inputs[ID], 0, -1)[..., np.newaxis]
            else:
                return self.inputs[ID][..., np.newaxis]
        else:
            if self.model_type == "RNN":
                return np.swapaxes(np.load(f"{self.data_filepath}{ID}.npy"), 0, -1)[..., np.newaxis]
            else:
                return np.load(f"{self.data_filepath}{ID}.npy")[..., np.newaxis]

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dimX, self.n_channels))
        y = np.empty((self.batch_size, self.dimY))

        for i, ID in enumerate(list_IDs_temp):           
            X[i,] = self.loadX(ID)
            y[i] = self.labels[ID]

        return X, y

    def close_file(self):
        if isinstance(self.file, io.TextIOWrapper):
            try:
                self.file.close()
            except:
                pass
        
    def extract_labels(self):
        """
        Extracting all the labels, used for testing to access true values of 'labels'
        """
        labels = []
        IDs = []
        for ID in self.list_IDs:
            IDs.append(ID)
            labels.append(self.labels[ID])
        labels = np.array(labels)
        return labels, IDs

class TimeHistory(keras.callbacks.Callback):
    def __init__(self, filename):
        self.filename = filename

    def on_train_begin(self, logs={}):
        self.file = open(self.filename, 'a')

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.file.write(f"{time.time() - self.epoch_time_start}\n")
        self.file.flush()
        os.fsync(self.file.fileno())
    def on_train_end(self, logs={}):
        self.file.close()


class LR_tracer(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        lr = keras.backend.eval( self.model.optimizer.lr )
        print(f"LR: {lr:.10f}")


class LR_scheduler:
    def __init__(self, total_epochs, inital_LR, multi_gpu_run = False, reduce_factor = 0.1):
        self.total_epochs = total_epochs
        self.initial_LR = inital_LR
        self.multi_gpu_run = multi_gpu_run
        self.reduce_factor = reduce_factor
    def scheduler(self, epoch):
        """
        Returns learning rate at a given epoch. 
        Recieves total number of epochs and initial learning rate
        """
        # print(f"IN LR_scheduler, initLR {self.initial_LR}, epoch {epoch}, frac. {(epoch + 1) / self.total_epochs}")
        if (epoch + 1) / self.total_epochs < 0.5:
            return self.initial_LR
        elif (epoch + 1) / self.total_epochs < 0.75:
            return self.initial_LR * self.reduce_factor
        else:
            return self.initial_LR * self.reduce_factor ** 2
    def callback(self):
        if self.multi_gpu_run == True:
            return hvd.callbacks.LearningRateScheduleCallback(self.scheduler)
        else:
            return tf.keras.callbacks.LearningRateScheduler(self.scheduler)


def R2(y_true, y_pred):
        SS_res = keras.backend.sum(keras.backend.square(y_true-y_pred)) 
        SS_tot = keras.backend.sum(keras.backend.square(y_true - keras.backend.mean(y_true, axis=0)))
        return (1 - SS_res/(SS_tot + keras.backend.epsilon()))
def R2_numpy(y_true, y_pred):
        SS_res = np.sum((y_true - y_pred)**2) 
        SS_tot = np.sum((y_true - np.mean(y_true))**2)
        return (1 - SS_res/(SS_tot + 1e-7))
def R2_final(y_true, y_pred):
        SS_res = np.sum((y_true - y_pred)**2)
        SS_tot = np.sum((y_true - np.mean(y_true, axis=0))**2)
        return (1 - SS_res/(SS_tot + 1e-7))


def define_callbacks():
    if ctx.inputs.gpus == 1:
        saving_callbacks = True
        horovod_callbacks = False
    else:
        saving_callbacks = True if hvd.rank() == 0 else False
        horovod_callbacks = True

    if saving_callbacks == True:
        saving_callbacks = [
            # keras.callbacks.TensorBoard(ctx.logdir, update_freq="epoch"),
            # hp.KerasCallback(logdir, HP_TensorBoard),
            TimeHistory(f"{ctx.filepath}_time.txt"),
            keras.callbacks.ModelCheckpoint(f"{ctx.filepath}_best.hdf5", monitor='val_loss', save_best_only=True, verbose=True),
            keras.callbacks.ModelCheckpoint(f"{ctx.filepath}_last.hdf5", monitor='val_loss', save_best_only=False, verbose=True), 
            keras.callbacks.CSVLogger(f"{ctx.filepath}.log", separator=',', append=True),
            LR_tracer(),
            ]
    else:
        saving_callbacks = []
    if horovod_callbacks == True:
        horovod_callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            ]
        if ctx.load_model == False:
            horovod_callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=ctx.inputs.warmup))
    else:
        horovod_callbacks = []
    
    important_callbacks = [
        keras.callbacks.TerminateOnNaN(),
        ]
    if ctx.HP.ReducingLR == True:
        scheduler = LR_scheduler(ctx.HP.MaxEpochs, ctx.HP.LearningRate, reduce_factor = 0.1)
        important_callbacks.append(scheduler.callback())

    #not saving into ctx as it might screw up things during broadcast of variables
    return horovod_callbacks + saving_callbacks + important_callbacks

def define_model(restore_training):
    model_exists = os.path.exists(f"{ctx.filepath}_last.hdf5")
    #define in what case to load the model
    # if model_exists == True and restore_training == True:
    #     if ctx.inputs.gpus == 1:
    #         load_model = True
    #         load_function = keras.models.load_model
    #     else:
    #         if hvd.rank() == 0:
    #             load_model = True
    #             load_function = hvd.load_model
    #         else:
    #             load_model = False
    # else:
    #     load_model = False
    if model_exists == True and restore_training == True:
        load_model = True
        if ctx.inputs.gpus == 1:
            load_function = keras.models.load_model
        else:
            load_function = hvd.load_model
    else:
        load_model = False
    ctx.load_model = load_model

    #load the model
    if load_model == True:
        custom_obj = {}
        custom_obj["R2"] = R2
        if ctx.HP.ActivationFunction[0] == "leakyrelu":
            custom_obj[ctx.HP.ActivationFunction[0]] = ctx.HP.ActivationFunction[1]["activation"]
        #if loading last model fails for some reason, load the best one
        try:
            ctx.model = load_function(f"{ctx.filepath}_last.hdf5", custom_objects=custom_obj)
        except:
            ctx.model = load_function(f"{ctx.filepath}_best.hdf5", custom_objects=custom_obj)

        with open(f"{ctx.filepath}.log") as f:
            number_of_epochs_trained = len(f.readlines()) - 1  #the first line is description
            print("NUMBER_OF_EPOCHS_TRAINED", number_of_epochs_trained)
        if ctx.HP.Epochs + number_of_epochs_trained > ctx.HP.MaxEpochs:
            final_epochs = ctx.HP.MaxEpochs
        else:
            final_epochs = ctx.HP.Epochs + number_of_epochs_trained

        ctx.fit_options = {
            "epochs": final_epochs,
            "initial_epoch": number_of_epochs_trained,
            "steps_per_epoch": ctx.Data.steps_per_epoch,
            "validation_steps": ctx.Data.validation_steps,
            }
        ctx.compile_options = {}
    else:
        ctx.fit_options = {
            "epochs": ctx.HP.Epochs,
            "initial_epoch": 0,
            "steps_per_epoch": ctx.Data.steps_per_epoch,
            "validation_steps": ctx.Data.validation_steps,
            }
        ctx.compile_options = {
            "loss": ctx.HP.Loss[1],
            "optimizer": ctx.HP.Optimizer[0](**ctx.HP.Optimizer[2]),
            "metrics": [R2],
            }
        if ctx.inputs.gpus > 1:
            ctx.compile_options["optimizer"] = hvd.DistributedOptimizer(ctx.compile_options["optimizer"])

def run_model(restore_training = True):
    #build callbacks and model
    callbacks = define_callbacks()
    define_model(restore_training)

    print(ctx.compile_options)
    print(ctx.fit_options)
    if len(ctx.compile_options) > 0:
        ctx.model.compile(**ctx.compile_options)

    verbose = ctx.inputs.verbose if ctx.main_process == True else 0
    # verbose = 2

    #fit model
    ctx.model.fit(
        ctx.Data.X["train"], 
        ctx.Data.Y["train"],
        batch_size=ctx.HP.BatchSize,
        callbacks=callbacks,
        validation_data=(ctx.Data.X["val"], ctx.Data.Y["val"]),
        verbose=verbose,
        **ctx.fit_options,
        )

    #predict on test data
    if ctx.main_process == True:
        true = ctx.Data.Y["test"]
        pred = ctx.model.predict(ctx.Data.X["test"], verbose=False)
        np.save(f"{ctx.filepath}_prediction_last.npy", pred)

        #making prediction from best model
        custom_obj = {}
        custom_obj["R2"] = R2
        #if activation is leakyrelu add to custom_obj
        if ctx.HP.ActivationFunction[0] == "leakyrelu":
            custom_obj[ctx.HP.ActivationFunction[0]] = ctx.HP.ActivationFunction[1]["activation"]
        ctx.model = keras.models.load_model(f"{ctx.filepath}_best.hdf5", custom_objects=custom_obj)
        
        pred = ctx.model.predict(ctx.Data.X["test"], verbose=False)
        np.save(f"{ctx.filepath}_prediction_best.npy", pred)

        with open(f"{ctx.filepath}_summary.txt", "w") as f:
            f.write(f"DATA: {str(ctx.Data)}\n")
            f.write(f"HYPARAMETERS: {str(ctx.HP)}\n")
            f.write(f"R2_total: {R2_final(true, pred)}\n")
            for i in range(4):
                print(f"R2: {R2_numpy(true[:, i], pred[:, i])}")
                f.write(f"R2_{i}: {R2_numpy(true[:, i], pred[:, i])}\n")
            f.write("\n")
            stringlist = []
            ctx.model.summary(print_fn=lambda x: stringlist.append(x))
            f.write("\n".join(stringlist))
    else:
        #task is killed if other processes finished and the above one on node 0 is still running
        with open(f"{ctx.filepath}_time.txt", "r") as f:
            for epoch_time in f:
                pass
        #wait for a fraction of epoch time
        time.sleep(float(epoch_time) * 0.2)
    
def run_large_model(restore_training = True):
    verbose = ctx.inputs.verbose if ctx.main_process == True else 0

    #build callbacks and model
    callbacks = define_callbacks()
    define_model(restore_training)
    if len(ctx.compile_options) > 0:
        ctx.model.compile(**ctx.compile_options)

    if ctx.inputs.tfrecord_database == True:
        ctx.model.fit(
            ctx.Data.train_ds,
            validation_data=ctx.Data.validation_ds,
            verbose = verbose,
            callbacks = callbacks,
            **ctx.fit_options,
            )        
    else:
        generator_options = {
            "labels": ctx.Data.labels, 
            "inputs": ctx.Data.inputs,
            "dimX": ctx.inputs.X_shape, 
            "dimY": ctx.inputs.Y_shape,
            "data_filepath": ctx.inputs.data_location,
            "model_type": ctx.inputs.model_type,
            "batch_size": ctx.HP.BatchSize,
            "load_all": ctx.inputs.load_all,
            }
        if ctx.inputs.noise_rolling == True:
            partition = {
                "train": ctx.Data.noise_rolling_partition["train"],
                # "validation": ctx.Data.partition["validation"],
                "validation": ctx.Data.noise_rolling_partition["validation"][0], #zeroth noise
                # "test": ctx.Data.partition["test"],
            }
        else:
            partition = {
                "train": ctx.Data.partition["train"],
                "validation": ctx.Data.partition["validation"],
                # "test": ctx.Data.partition["test"],
            }
        ctx.generators = {
            "train": DataGenerator(partition["train"], **generator_options, initial_epoch = ctx.fit_options["initial_epoch"], N_noise = ctx.inputs.N_noise, noise_rolling = ctx.inputs.noise_rolling, iterations = ctx.fit_options["steps_per_epoch"]),
            "validation": SimpleDataGenerator(partition["validation"], **generator_options, iterations = ctx.fit_options["validation_steps"]),
            # "test": SimpleDataGenerator(partition["test"], **generator_options, data_type = "test"),
            }
        
        workers = 1 if ctx.inputs.load_all == True else ctx.inputs.workers
        use_multiprocessing = False if ctx.inputs.load_all == True else True
        # verbose = 2

        #fit model
        ctx.model.fit(
            ctx.generators["train"],
            validation_data=ctx.generators["validation"],
            verbose = verbose,
            max_queue_size = 16,
            use_multiprocessing = use_multiprocessing,
            workers = workers,
            callbacks = callbacks,
            **ctx.fit_options,
            )

def predict_large(Type):
    """
    Type: "best" or "last"
    """
    custom_obj = {}
    custom_obj["R2"] = R2
    #if activation is leakyrelu add to custom_obj
    if ctx.HP.ActivationFunction[0] == "leakyrelu":
        custom_obj[ctx.HP.ActivationFunction[0]] = ctx.HP.ActivationFunction[1]["activation"]
    ctx.model = keras.models.load_model(f"{ctx.filepath}_{Type}.hdf5", custom_objects=custom_obj)
    print(f"PREDICTING THE MODEL {Type}")

    if ctx.inputs.tfrecord_database == True:
        # assumes eager execution
        true = []
        for x, y in ctx.Data.test_ds:
            true.append(y.numpy())
        true = np.concatenate(true, axis = 0)

        pred = ctx.model.predict(
            ctx.Data.test_ds,
            verbose = False,
            )
    else:
        generator_options = {
            "labels": ctx.Data.labels, 
            "inputs": ctx.Data.inputs,
            "dimX": ctx.inputs.X_shape, 
            "dimY": ctx.inputs.Y_shape,
            "data_filepath": ctx.inputs.data_location,
            "model_type": ctx.inputs.model_type,
            "batch_size": ctx.HP.BatchSize,
            "load_all": ctx.inputs.load_all,
            }
        generator = SimpleDataGenerator(ctx.Data.partition["test"], **generator_options, filename = f"{ctx.filepath}_true_{Type}.txt")
        
        workers = 1 if ctx.inputs.load_all == True else ctx.inputs.workers
        use_multiprocessing = False if ctx.inputs.load_all == True else True
        
        pred = ctx.model.predict(
            generator, 
            max_queue_size = 512, 
            workers = workers, 
            use_multiprocessing = use_multiprocessing,
            verbose = False,
            )
        generator.close_file()
        # print(pred)
        # np.save(f"{ctx.filepath}_prediction_{Type}.npy", pred)
        true = []
        with open(f"{ctx.filepath}_true_{Type}.txt", "r") as f:
            for line in f:
                print(line, end="")
                true.append([float(i) for i in line.rstrip("\n").split(" ")[1:]])
        true = np.array(true)

    np.save(f"{ctx.filepath}_prediction_{Type}.npy", pred)
    np.save(f"{ctx.filepath}_true_{Type}.npy", true)

    R2_score = []
    R2_score.append(R2_final(true, pred))
    for i in range(4):
        R2_score.append(R2_numpy(true[:, i], pred[:, i]))

    if Type == "best":
        with open(f"{ctx.filepath}_summary.txt", "w") as f:
            f.write(f"DATA: {str(ctx.Data)}\n")
            f.write(f"HYPARAMETERS: {str(ctx.HP)}\n")
            f.write(f"R2_total: {R2_score[0]}\n")
            for i in range(4):
                print(f"R2: {R2_score[i+1]}")
                f.write(f"R2_{i}: {R2_score[i+1]}\n")
            f.write("\n")
            stringlist = []
            ctx.model.summary(print_fn=lambda x: stringlist.append(x))
            f.write("\n".join(stringlist))

    return true, pred, R2_score