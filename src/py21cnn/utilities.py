import os
import time
import copy
import hashlib
import numpy as np
import tensorflow as tf
# import keras
from tensorflow import keras
# import horovod.tensorflow.keras as hvd

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
        ):
        self.filepath = filepath
        self.dimensionality = dimensionality
        self.removed_average = removed_average
        self.normalized = normalized
        self.Zmax = Zmax
        self.filetype = filetype
        if len(formatting) == 0:
            default_formatting = ['clipped_-250_+50', 'NaN_removed']
            if self.dimensionality == 2:
                default_formatting.append('boxcar44')
                default_formatting.append('10slices')
            if self.dimensionality == 3:
                default_formatting.append('boxcar444')
                default_formatting.append('sliced22')
            default_formatting.sort()
            self.formatting = default_formatting
        else:
            formatting.sort()
            self.formatting = formatting

    def __str__(self):
        S = f"dim:{self.dimensionality}__removed_average:{self.removed_average}__normalized:{self.normalized}__Zmax:{self.Zmax}__dtype:{self.filetype}"
        for i in self.formatting:
            S += f"__{i}"
        return S
    
    def hash(self):
        return hashlib.md5(self.__str__().encode()).hexdigest()

    def loadTVT(self, model_type = None, pTVT = [0.8, 0.1, 0.1]):
        self.X = {}
        self.Y = {}
        Hash = self.hash()
        for p, key in zip(pTVT, ['train', 'val', 'test']):
            self.X[key] = np.load(f"{self.filepath}X_{key}_{p:.2f}_{Hash}.npy")
            if model_type == "RNN":
                self.X[key] = np.swapaxes(self.X[key], -1, -self.dimensionality) #swapping to get time dimension on right place
            self.X[key] = self.X[key][..., np.newaxis]
            self.Y[key] = np.load(f"{self.filepath}Y_{key}_{p:.2f}_{Hash}.npy")
        self.shape = self.X['test'].shape[1:]
        # return self.X, self.Y


class AuxiliaryHyperparameters:
    def __init__(
        self,
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
        Epochs = 200,
        ):
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

    def __str__(self):
        S = f"Loss:{self.Loss[1]}__Optimizer:{self.Optimizer[1]}__LR:{self.LearningRate:.10f}__Activation:{self.ActivationFunction[0]}"
        S += f"__BN:{self.BatchNormalization}__dropout:{self.Dropout:.2f}__reduceLR:{self.ReducingLR}__Batch:{self.BatchSize:05d}__Epochs:{self.Epochs:05d}"
        return S
    def hashstring(self):
        #differences from __str__ in not including epochs
        S = f"Loss:{self.Loss[1]}__Optimizer:{self.Optimizer[1]}__LR:{self.LearningRate:.10f}__Activation:{self.ActivationFunction[0]}"
        S += f"__BN:{self.BatchNormalization}__dropout:{self.Dropout:.2f}__reduceLR:{self.ReducingLR}__Batch:{self.BatchSize:05d}"
        return S
    def hash(self):
        return hashlib.md5(self.hashstring().encode()).hexdigest()




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
        print(' LR: %.10f '%(lr))


def R2(y_true, y_pred):
        SS_res = keras.backend.sum(keras.backend.square(y_true-y_pred)) 
        SS_tot = keras.backend.sum(keras.backend.square(y_true - keras.backend.mean(y_true)))
        return (1 - SS_res/(SS_tot + keras.backend.epsilon()))
def R2_numpy(y_true, y_pred):
        SS_res = np.sum((y_true - y_pred)**2) 
        SS_tot = np.sum((y_true - np.mean(y_true))**2)
        return (1 - SS_res/(SS_tot + 1e-7))
def R2_final(y_true, y_pred):
        SS_res = np.sum((y_true - y_pred)**2)
        SS_tot = np.sum((y_true - np.mean(y_true, axis=0))**2)
        return (1 - SS_res/(SS_tot + 1e-7))

def run_multigpu_model(model, Data, AuxHP, HP, inputs, hvd):

    filepath = f"{inputs.saving_location}{inputs.file_prefix}{inputs.model[0]}_{inputs.model[1]}_{HP.hash()}_{Data.hash()}"
    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        # LR_tracer(),
        keras.callbacks.TerminateOnNaN(),
        # keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=35, verbose=True),
        ]
    if AuxHP.ReducingLR == True:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=True))
    if hvd.rank() == 0:
        callbacks.append(TimeHistory(f"{filepath}_time.txt")),
        callbacks.append(keras.callbacks.ModelCheckpoint(f"{filepath}_best.hdf5", monitor='val_loss', save_best_only=True, verbose=True)),
        callbacks.append(keras.callbacks.ModelCheckpoint(f"{filepath}_last.hdf5", monitor='val_loss', save_best_only=False, verbose=True)), 
        callbacks.append(keras.callbacks.CSVLogger(f"{filepath}.log", separator=',', append=True)),

    model.compile(  loss=AuxHP.Loss[1],
                    optimizer=hvd.DistributedOptimizer(AuxHP.Optimizer[0](**AuxHP.Optimizer[2])),
                    metrics = [R2]
                    )

    history = model.fit(Data.X['train'], Data.Y['train'],
                        epochs=AuxHP.Epochs,
                        batch_size=AuxHP.BatchSize,
                        callbacks=callbacks,
                        validation_data=(Data.X['val'], Data.Y['val']),
                        verbose=2,
                        )
    
    if hvd.rank() == 0:
        prediction = model.predict(Data.X['test'], verbose=False)
        np.save(f"{filepath}_prediction.npy", prediction)

        with open(f"{filepath}_summary.txt", "w") as f:
            f.write(f"DATA: {str(Data)}\n")
            f.write(f"HYPARAMETERS: {str(HP)}\n")
            # f.write(f"R2_total: {R2_numpy(Data.Y['test'], prediction)}\n")
            for i in range(4):
                print(f"R2: {R2_numpy(Data.Y['test'][:, i], prediction[:, i])}")
                f.write(f"R2_{i}: {R2_numpy(Data.Y['test'][:, i], prediction[:, i])}\n")
            f.write("\n")
            # f.write(model.summary())
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            f.write("\n".join(stringlist))





def run_model(model, Data, AuxHP, inputs):

    filepath = f"{inputs.saving_location}{inputs.file_prefix}{inputs.model[0]}_{inputs.model[1]}_{AuxHP.hash()}_{Data.hash()}"
    
    callbacks = [
        LR_tracer(),
        TimeHistory(f"{filepath}_time.txt"),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(f"{filepath}_best.hdf5", monitor='val_loss', save_best_only=True, verbose=True),
        keras.callbacks.ModelCheckpoint(f"{filepath}_last.hdf5", monitor='val_loss', save_best_only=False, verbose=True), 
        keras.callbacks.CSVLogger(f"{filepath}.log", separator=',', append=True),
        # keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=35, verbose=True),
    ]
    if AuxHP.ReducingLR == True:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=True))

    # if the model has been run before, load it and run again for AuxHP.Epoch - number of epochs from before
    # else, compile the model and run it
    if os.path.exists(f"{filepath}_last.hdf5") == True:
        custom_obj = {}
        custom_obj["R2"] = R2
        # custom_obj["TimeHistory"] = TimeHistory
        #if activation is leakyrelu add to custom_obj
        if AuxHP.ActivationFunction[0] == "leakyrelu":
            custom_obj[AuxHP.ActivationFunction[0]] = AuxHP.ActivationFunction[1]["activation"]
        #if loading last model fails for some reason, load the best one
        try:
            model = keras.models.load_model(f"{filepath}_last.hdf5", custom_objects=custom_obj)
        except:
            model = keras.models.load_model(f"{filepath}_best.hdf5", custom_objects=custom_obj)
        model.summary()
        
        with open(f"{filepath}.log") as f:
            number_of_lines = len(f.readlines())
            number_of_epochs_trained = number_of_lines - 1  #the first line is description
            print(number_of_epochs_trained)
            if number_of_epochs_trained >= AuxHP.Epochs:
                print(number_of_epochs_trained, ">=", AuxHP.Epochs)
                raise ValueError('number_of_epochs_trained >= AuxiliaryHyperparameters.Epochs')

            model.evaluate(Data.X['train'], Data.Y['train'], verbose=False)
            model.evaluate(Data.X['val'], Data.Y['val'], verbose=False)

            history = model.fit(Data.X['train'], Data.Y['train'],
                                initial_epoch=number_of_epochs_trained,
                                epochs=AuxHP.Epochs,
                                batch_size=AuxHP.BatchSize,
                                callbacks=callbacks,
                                validation_data=(Data.X['val'], Data.Y['val']),
                                verbose=2,
                                )
    else:
        model.compile(  loss=AuxHP.Loss[1],
                        optimizer=AuxHP.Optimizer[0](**AuxHP.Optimizer[2]),
                        metrics = [R2])


        history = model.fit(Data.X['train'], Data.Y['train'],
                            epochs=AuxHP.Epochs,
                            batch_size=AuxHP.BatchSize,
                            callbacks=callbacks,
                            validation_data=(Data.X['val'], Data.Y['val']),
                            verbose=2,
                            )
    
    prediction = model.predict(Data.X['test'], verbose=False)
    np.save(f"{filepath}_prediction.npy", prediction)

    with open(f"{filepath}_summary.txt", "w") as f:
        f.write(f"DATA: {str(Data)}\n")
        f.write(f"HYPARAMETERS: {str(AuxHP)}\n")
        # f.write(f"R2_total: {R2_numpy(Data.Y['test'], prediction)}\n")
        for i in range(4):
            print(f"R2: {R2_numpy(Data.Y['test'][:, i], prediction[:, i])}")
            f.write(f"R2_{i}: {R2_numpy(Data.Y['test'][:, i], prediction[:, i])}\n")
        f.write("\n")
        # f.write(model.summary())
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        f.write("\n".join(stringlist))