from . import ctx

from .formatting import Filters
import os
import time
import copy
import hashlib
import numpy as np
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

    def loadTVT(self, model_type = None, pTVT = [0.8, 0.1, 0.1]):
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
        ctx,
        dimensionality = 2,
        removed_average = True,
        normalized = True,
        Zmax = 30,
        filetype = 'float32',
        formatting = [],
        noise = ["tools21cm", "SKA1000"],
        shape = None,
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
        self.load(ctx)

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

    def load(self, ctx):
        permutation = Filters.constructIndexArray(ctx.inputs.N_walker, *ctx.inputs.pTVT, 1312)
        # print(permutation)
        Y = np.load(f"{ctx.inputs.data_location}{ctx.inputs.Y_filename}.npy")
        ctx.inputs.Y_shape = Y.shape[-1]
        self.partition = {
            "train": [], 
            "validation": [], 
            "test": []}
        self.noise_rolling_partition = {
            "train": [], 
            "validation": [], 
            "test": []}
        for key in list(self.noise_rolling_partition.keys()):
            for seed in range(ctx.inputs.N_noise):
                self.noise_rolling_partition[key].append([])
        self.labels = {}
        keys = list(self.partition.keys())
        for walker in range(ctx.inputs.N_walker):
            for s in range(ctx.inputs.N_slice):
                for seed in range(ctx.inputs.N_noise):
                    self.partition[keys[permutation[walker]]].append(ctx.inputs.X_fstring.format(walker, s, seed))
                    self.noise_rolling_partition[keys[permutation[walker]]][seed].append(ctx.inputs.X_fstring.format(walker, s, seed))
                    self.labels[ctx.inputs.X_fstring.format(walker, s, seed)] = Y[walker]
        # print(self.partition)



class AuxiliaryHyperparameters:
    def __init__(
        self,
        model_name,
        Epochs = 200,
        MaxEpochs = 200,
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

    def __str__(self):
        S = f"Loss:{self.Loss[1]}__Optimizer:{self.Optimizer[1]}__LR:{self.LearningRate:.10f}__Activation:{self.ActivationFunction[0]}"
        S += f"__BN:{self.BatchNormalization}__dropout:{self.Dropout:.2f}__reduceLR:{self.ReducingLR}__Batch:{self.BatchSize:05d}__Epochs:{self.MaxEpochs:05d}"
        return S
    def hashstring(self):
        #differences from __str__ in not including epochs
        S = f"Loss:{self.Loss[1]}__Optimizer:{self.Optimizer[1]}__LR:{self.LearningRate:.10f}__Activation:{self.ActivationFunction[0]}"
        S += f"__BN:{self.BatchNormalization}__dropout:{self.Dropout:.2f}__reduceLR:{self.ReducingLR}__Batch:{self.BatchSize:05d}"
        return S
    def hash(self):
        return hashlib.md5(self.hashstring().encode()).hexdigest()


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                list_IDs,
                labels, 
                dimX, 
                dimY,
                data_filepath,
                model_type,
                batch_size, 
                initial_epoch,
                N_noise,
                noise_rolling,
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
        self.labels = labels
        self.list_IDs = list_IDs
        self.N_noise = N_noise
        self.noise_index = initial_epoch % self.N_noise - 1
        self.noise_rolling = noise_rolling
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.iterations = self.__len__()
        self.iteration_index = 0
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.noise_rolling == True:
            return len(self.list_IDs[0]) // self.batch_size
        else:
            return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        #define noise_index
        self.iteration_index = (self.iteration_index + 1) % self.iterations
        if self.iteration_index == 0:
            self.noise_index = (self.noise_index + 1) % self.N_noise
        # Find list of IDs
        if self.noise_rolling == True:
            list_IDs_temp = [self.list_IDs[self.noise_index][k] for k in indexes]
        else:
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        print(list_IDs_temp)

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        # return tf.data.Dataset.from_tensor_slices((X, y))
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.noise_rolling == True:
            # self.noise_index = (self.noise_index + 1) % self.N_noise
            self.indexes = np.arange(len(self.list_IDs[0]))
        else:
            self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def loadX(self, filename):
        if self.model_type == "RNN":
            return np.swapaxes(np.load(filename), 0, -1)[..., np.newaxis]
        else:
            return np.load(filename)[..., np.newaxis]
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dimX, self.n_channels))
        y = np.empty((self.batch_size, self.dimY))

        for i, ID in enumerate(list_IDs_temp):
            X[i,] = self.loadX(f"{self.data_filepath}{ID}.npy")
            y[i] = self.labels[ID]

        return X, y

    def extract_labels(self):
        """
        Extracting all the labels, used for testing purposes to access true values of 'labels'
        """
        y = np.empty((len(self.list_IDs), self.dimY))
        for i, ID in enumerate(self.list_IDs):
            y[i] = self.labels[ID]
        return y, self.list_IDs


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
            keras.callbacks.TensorBoard(ctx.logdir, update_freq='batch'),
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
            hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=ctx.inputs.warmup),
            ]
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

    #determine steps_per_epoch
    if isinstance(ctx.Data, Data):
        steps_per_epoch = ctx.Data.X["train"].shape[0] // ctx.inputs.gpus // ctx.HP.BatchSize
        validation_steps = ctx.Data.X["val"].shape[0] // ctx.HP.BatchSize
    elif isinstance(ctx.Data, LargeData):
        if ctx.inputs.noise_rolling == True:
            steps_per_epoch = len(ctx.Data.noise_rolling_partition["train"][0]) // ctx.inputs.gpus // ctx.HP.BatchSize
            validation_steps = len(ctx.Data.noise_rolling_partition["validation"][0]) // ctx.HP.BatchSize   
        else:         
            steps_per_epoch = len(ctx.Data.partition["train"]) // ctx.inputs.gpus // ctx.HP.BatchSize
            validation_steps = len(ctx.Data.partition["validation"]) // ctx.HP.BatchSize
    else:
        raise TypeError("ctx.Data should be an instance of {Data, LargeData} class")

    if validation_steps == 0:
        raise ValueError("Number of validation steps per epoch is 0. Change batch size, validation probability, or give me more data.")

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
            "steps_per_epoch": steps_per_epoch,
            "validation_steps": validation_steps,
            }
        ctx.compile_options = {}
    else:
        ctx.fit_options = {
            "epochs": ctx.HP.Epochs,
            "initial_epoch": 0,
            "steps_per_epoch": steps_per_epoch,
            "validation_steps": validation_steps,
            }
        ctx.compile_options = {
            "loss": ctx.HP.Loss[1],
            "optimizer": ctx.HP.Optimizer[0](**ctx.HP.Optimizer[2]),
            "metrics": [R2],
            }
        if ctx.inputs.gpus > 1:
            ctx.compile_options["optimizer"] = hvd.DistributedOptimizer(ctx.compile_options["optimizer"])

def run_model(restore_training = True):
    #build callbacks
    callbacks = define_callbacks()
    define_model(restore_training)
    if len(ctx.compile_options) > 0:
        ctx.model.compile(**ctx.compile_options)

    verbose = 2 if ctx.main_process == True else 0
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
    #build callbacks
    callbacks = define_callbacks()
    define_model(restore_training)
    if len(ctx.compile_options) > 0:
        ctx.model.compile(**ctx.compile_options)

    generator_options = {
        "labels": ctx.Data.labels, 
        "dimX": ctx.inputs.X_shape, 
        "dimY": ctx.inputs.Y_shape,
        "data_filepath": ctx.inputs.data_location,
        "model_type": ctx.inputs.model_type,
        "batch_size": ctx.HP.BatchSize, 
        "initial_epoch": ctx.fit_options["initial_epoch"],
        "N_noise": ctx.inputs.N_noise,
        }
    if ctx.inputs.noise_rolling == True:
        data_partition = ctx.Data.noise_rolling_partition
    else:
        data_partition = ctx.Data.partition
    ctx.generators = {
        "train": DataGenerator(data_partition["train"], **generator_options, noise_rolling = ctx.inputs.noise_rolling),
        "validation": DataGenerator(data_partition["validation"], **generator_options, noise_rolling = ctx.inputs.noise_rolling),
        "test": DataGenerator(ctx.Data.partition["test"], **generator_options, shuffle = False, noise_rolling = False),
        }
    
    verbose = 2 if ctx.main_process == True else 0
    # verbose = 2

    #fit model
    ctx.model.fit(
        ctx.generators["train"],
        validation_data=ctx.generators["validation"],
        verbose = verbose,
        max_queue_size = 16,
        use_multiprocessing = True,
        workers = ctx.inputs.workers,
        callbacks = callbacks,
        **ctx.fit_options,
        )

    #predict on test data
    if ctx.main_process == True:
        print("PREDICTING THE MODEL")
        true, IDs = ctx.generators["test"].extract_labels()
        print(IDs)
        pred = ctx.model.predict(
            ctx.generators["test"], 
            max_queue_size = 16, 
            workers = ctx.inputs.workers, 
            use_multiprocessing = True,
            verbose = False,
            )
        print(true)
        print(pred)
        np.save(f"{ctx.filepath}_prediction_last.npy", pred)

        #making prediction from best model
        custom_obj = {}
        custom_obj["R2"] = R2
        #if activation is leakyrelu add to custom_obj
        if ctx.HP.ActivationFunction[0] == "leakyrelu":
            custom_obj[ctx.HP.ActivationFunction[0]] = ctx.HP.ActivationFunction[1]["activation"]
        ctx.model = keras.models.load_model(f"{ctx.filepath}_best.hdf5", custom_objects=custom_obj)
        pred = ctx.model.predict(
            ctx.generators["test"], 
            max_queue_size = 16, 
            workers = ctx.inputs.workers, 
            use_multiprocessing = True,
            verbose = False,
            )
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






















# def run_multigpu_model(model, Data, AuxHP, HP, HP_TensorBoard, inputs, restore_weights = True, restore_training = True, warmup=False):

#     filepath = f"{inputs.saving_location}{inputs.file_prefix}{inputs.model[0]}_{inputs.model[1]}_{HP.hash()}_{Data.hash()}"
#     logdir = f"{inputs.logs_location}{inputs.file_prefix}{inputs.model[0]}/{inputs.model[1]}/{Data.hash()}/{HP.hash()}"
    
#     callbacks = []
#     callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
#     callbacks.append(hvd.callbacks.MetricAverageCallback())
#     callbacks.append(keras.callbacks.TerminateOnNaN())
#     if warmup == True:
#         callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=inputs.patience)) # patience in ReduceLROnPlateau and warmup_epochs should be the same order of magnitude, therefore we choose the same value
#     # if AuxHP.ReducingLR == True:
#     #     callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=inputs.patience, verbose=True))
#     if hvd.rank() == 0:
#         callbacks.append(TimeHistory(f"{filepath}_time.txt"))
#         callbacks.append(keras.callbacks.ModelCheckpoint(f"{filepath}_best.hdf5", monitor='val_loss', save_best_only=True, verbose=True))
#         callbacks.append(keras.callbacks.ModelCheckpoint(f"{filepath}_last.hdf5", monitor='val_loss', save_best_only=False, verbose=True))
#         callbacks.append(keras.callbacks.CSVLogger(f"{filepath}.log", separator=',', append=True))
#         #callbacks.append(keras.callbacks.TensorBoard(logdir, histogram_freq = 1, batch_size=AuxHP.BatchSize, write_graph=True, write_grads=True, write_images=True, update_freq=int(Data.TrainExamples//hvd.size())))
#         callbacks.append(keras.callbacks.TensorBoard(logdir, update_freq='batch'))
#         #callbacks.append(hp.KerasCallback(logdir, HP_TensorBoard))
#         # manually writing hyperparameters instead of calling keras callback
#         # with tf.compat.v1.create_file_writer(logdir).as_default() as w:
#         #     sess.run(w.init())
#         #     sess.run(hp.hparams(HP_TensorBoard))
#         #     sess.run(w.flush())
#         # with tf.summary.FileWriter(logdir) writer:
#     if AuxHP.ReducingLR == True:
#         scheduler = LR_scheduler(AuxHP.MaxEpochs, AuxHP.LearningRate, multi_gpu_run = True, reduce_factor = 0.1)
#         callbacks.append(scheduler.callback())

#     #deciding how to run a model: restore_weights and restore_training defines what happens
#     #if restore_weigths == True and there is a model to load then it loads it, recompiles only if restore_training == False
#     #in any other case it just compiles model that is sent to the function call, and runs it.
#     #for example, in case of restore_weights == False, restore_training is ignored, as it doesn't have meaning
#     if os.path.exists(f"{filepath}_last.hdf5") == True and restore_weights == True:
#         with open(f"{filepath}.log") as f:
#             number_of_lines = len(f.readlines())
#             number_of_epochs_trained = number_of_lines - 1  #the first line is description
#             print(number_of_epochs_trained)
#         if AuxHP.Epochs + number_of_epochs_trained > AuxHP.MaxEpochs:
#             final_epochs = AuxHP.MaxEpochs
#         else:
#             final_epochs = AuxHP.Epochs + number_of_epochs_trained
#         #broadcast final_epochs to all workers, I have no idea if that is necessary.
#         #but that's what they do in example https://github.com/horovod/horovod/blob/87ad738d4d6b14ffdcc55a03acdc3cfb03f380c8/examples/keras_imagenet_resnet50.py
#         final_epochs = hvd.broadcast(final_epochs, 0, name='final_epochs')
#         number_of_epochs_trained = hvd.broadcast(number_of_epochs_trained, 0, name='number_of_epochs_trained')
        
#         custom_obj = {}
#         custom_obj["R2"] = R2
#         #if activation is leakyrelu add to custom_obj
#         if AuxHP.ActivationFunction[0] == "leakyrelu":
#             custom_obj[AuxHP.ActivationFunction[0]] = AuxHP.ActivationFunction[1]["activation"]
#         #defining Optimizer name used for loading
#         if AuxHP.Optimizer[1] == "Momentum":
#             OptName = "SGD"
#         else:
#             OptName = AuxHP.Optimizer[1]
#         if restore_training == True:
#             custom_obj[OptName] = lambda **kwargs: hvd.DistributedOptimizer(AuxHP.Optimizer[0](**kwargs))
#         else:
#             custom_obj[OptName] =  hvd.DistributedOptimizer(AuxHP.Optimizer[0](**AuxHP.Optimizer[2]))

#         #if loading last model fails for some reason, load the best one
#         try:
#             model = keras.models.load_model(f"{filepath}_last.hdf5", custom_objects=custom_obj)
#         except:
#             model = keras.models.load_model(f"{filepath}_best.hdf5", custom_objects=custom_obj)

#         # if restore_training == True:
#         #     opt = model.optimizer
#         #     model.optimizer = hvd.DistributedOptimizer(opt)
#         #     model.optimizer.iterations = opt.iterations
#         #     model.optimizer.weights = opt.weights
#         # else:
#         #     model.optimizer = hvd.DistributedOptimizer(AuxHP.Optimizer[0](**AuxHP.Optimizer[2]))
        
#         # model.compile(  loss=AuxHP.Loss[1],
#         #                 optimizer=model.optimizer,
#         #                 metrics = [R2],
#         #                 experimental_run_tf_function=False,
#         #                 )
#             # callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=inputs.patience, verbose=True))

#         model.fit(  Data.X['train'], Data.Y['train'],
#                     initial_epoch=number_of_epochs_trained,
#                     epochs=final_epochs,
#                     batch_size=AuxHP.BatchSize,
#                     callbacks=callbacks,
#                     validation_data=(Data.X['val'], Data.Y['val']),
#                     verbose=2,
#                     )
#     else:
#             # callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=inputs.patience, verbose=True))
        
#         model.compile(  loss=AuxHP.Loss[1],
#                         optimizer=hvd.DistributedOptimizer(AuxHP.Optimizer[0](**AuxHP.Optimizer[2])),
#                         metrics = [R2],
#                         #experimental_run_tf_function=False,
#                         )

#         model.fit(  Data.X['train'], Data.Y['train'],
#                     epochs=AuxHP.Epochs,
#                     batch_size=AuxHP.BatchSize,
#                     callbacks=callbacks,
#                     validation_data=(Data.X['val'], Data.Y['val']),
#                     verbose=2,
#                     )
    
#     if hvd.rank() == 0:
#         prediction = model.predict(Data.X['test'], verbose=False)
#         np.save(f"{filepath}_prediction_last.npy", prediction)
#         #making prediction from best model
#         custom_obj = {}
#         custom_obj["R2"] = R2
#         #if activation is leakyrelu add to custom_obj
#         if AuxHP.ActivationFunction[0] == "leakyrelu":
#             custom_obj[AuxHP.ActivationFunction[0]] = AuxHP.ActivationFunction[1]["activation"]
#         model = keras.models.load_model(f"{filepath}_best.hdf5", custom_objects=custom_obj)
#         prediction = model.predict(Data.X['test'], verbose=False)
#         np.save(f"{filepath}_prediction_best.npy", prediction)

#         with open(f"{filepath}_summary.txt", "w") as f:
#             f.write(f"DATA: {str(Data)}\n")
#             f.write(f"HYPARAMETERS: {str(HP)}\n")
#             f.write(f"R2_total: {R2_final(Data.Y['test'], prediction)}\n")
#             for i in range(4):
#                 print(f"R2: {R2_numpy(Data.Y['test'][:, i], prediction[:, i])}")
#                 f.write(f"R2_{i}: {R2_numpy(Data.Y['test'][:, i], prediction[:, i])}\n")
#             f.write("\n")
#             # f.write(model.summary())
#             stringlist = []
#             model.summary(print_fn=lambda x: stringlist.append(x))
#             f.write("\n".join(stringlist))
# #    else:
# #        #task is killed if other processes finished and the above one on node 0 is still running
# #        time.sleep(300)



# def run_model(model, Data, AuxHP, HP_TensorBoard, inputs, restore_weights = True, restore_training = True):

#     filepath = f"{inputs.saving_location}{inputs.file_prefix}{inputs.model[0]}_{inputs.model[1]}_{AuxHP.hash()}_{Data.hash()}"
#     logdir = f"{inputs.logs_location}{inputs.file_prefix}{inputs.model[0]}/{inputs.model[1]}/{Data.hash()}/{AuxHP.hash()}"

#     callbacks = [
#         keras.callbacks.TensorBoard(logdir, update_freq='epoch'),
#         # hp.KerasCallback(logdir, HP_TensorBoard),
#         LR_tracer(),
#         TimeHistory(f"{filepath}_time.txt"),
#         keras.callbacks.TerminateOnNaN(),
#         keras.callbacks.ModelCheckpoint(f"{filepath}_best.hdf5", monitor='val_loss', save_best_only=True, verbose=True),
#         keras.callbacks.ModelCheckpoint(f"{filepath}_last.hdf5", monitor='val_loss', save_best_only=False, verbose=True), 
#         keras.callbacks.CSVLogger(f"{filepath}.log", separator=',', append=True),
#         # keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=35, verbose=True),
#     ]
#     # manually writing hyperparameters instead of calling keras callback
#     # with tf.compat.v2.create_file_writer(logdir).as_default() as w:
#     #     sess.run(w.init())
#     #     sess.run(hp.hparams(HP_TensorBoard))
#     #     sess.run(w.flush())
    

#     # if the model has been run before, load it and run again
#     # else, compile the model and run it
#     if os.path.exists(f"{filepath}_last.hdf5") == True and restore_weights == True:
#         custom_obj = {}
#         custom_obj["R2"] = R2
#         # custom_obj["TimeHistory"] = TimeHistory
#         #if activation is leakyrelu add to custom_obj
#         if AuxHP.ActivationFunction[0] == "leakyrelu":
#             custom_obj[AuxHP.ActivationFunction[0]] = AuxHP.ActivationFunction[1]["activation"]
#         #if loading last model fails for some reason, load the best one
#         try:
#             model = keras.models.load_model(f"{filepath}_last.hdf5", custom_objects=custom_obj)
#         except:
#             model = keras.models.load_model(f"{filepath}_best.hdf5", custom_objects=custom_obj)
#         model.summary()

#         with open(f"{filepath}.log") as f:
#             number_of_lines = len(f.readlines())
#             number_of_epochs_trained = number_of_lines - 1  #the first line is description
#             print("NUMBER_OF_EPOCHS_TRAINED", number_of_epochs_trained)
#             # if number_of_epochs_trained >= AuxHP.Epochs:
#             #     print(number_of_epochs_trained, ">=", AuxHP.Epochs)
#             #     raise ValueError('number_of_epochs_trained >= AuxiliaryHyperparameters.Epochs')
#         if AuxHP.Epochs + number_of_epochs_trained > AuxHP.MaxEpochs:
#             final_epochs = AuxHP.MaxEpochs
#         else:
#             final_epochs = AuxHP.Epochs + number_of_epochs_trained     
        
        
#         if AuxHP.ReducingLR == True:
#             scheduler = LR_scheduler(AuxHP.MaxEpochs, keras.backend.get_value(model.optimizer.lr), reduce_factor = 0.1)
#             callbacks.append(scheduler.callback())
#             # callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=inputs.patience, verbose=True))
        
#         #in the case we don't want to restore training, we recompile the model
#         if restore_training == False:
#             model.compile(  loss=AuxHP.Loss[1],
#                             optimizer=AuxHP.Optimizer[0](**AuxHP.Optimizer[2]),
#                             metrics = [R2])

#         model.fit(  Data.X['train'], Data.Y['train'],
#                     initial_epoch=number_of_epochs_trained,
#                     epochs=final_epochs,
#                     batch_size=AuxHP.BatchSize,
#                     callbacks=callbacks,
#                     validation_data=(Data.X['val'], Data.Y['val']),
#                     verbose=2,
#                             )
#     else:
#         if AuxHP.ReducingLR == True:
#             scheduler = LR_scheduler(AuxHP.MaxEpochs, AuxHP.LearningRate, reduce_factor = 0.1)
#             callbacks.append(scheduler.callback())
        
#         model.compile(  loss=AuxHP.Loss[1],
#                         optimizer=AuxHP.Optimizer[0](**AuxHP.Optimizer[2]),
#                         metrics = [R2])

#         model.fit(  Data.X['train'], Data.Y['train'],
#                     epochs=AuxHP.Epochs,
#                     batch_size=AuxHP.BatchSize,
#                     callbacks=callbacks,
#                     validation_data=(Data.X['val'], Data.Y['val']),
#                     verbose=2,
#                     )
    
#     prediction = model.predict(Data.X['test'], verbose=False)
#     np.save(f"{filepath}_prediction.npy", prediction)

#     with open(f"{filepath}_summary.txt", "w") as f:
#         f.write(f"DATA: {str(Data)}\n")
#         f.write(f"HYPARAMETERS: {str(AuxHP)}\n")
#         # f.write(f"R2_total: {R2_numpy(Data.Y['test'], prediction)}\n")
#         for i in range(4):
#             print(f"R2: {R2_numpy(Data.Y['test'][:, i], prediction[:, i])}")
#             f.write(f"R2_{i}: {R2_numpy(Data.Y['test'][:, i], prediction[:, i])}\n")
#         f.write("\n")
#         # f.write(model.summary())
#         stringlist = []
#         model.summary(print_fn=lambda x: stringlist.append(x))
#         f.write("\n".join(stringlist))

