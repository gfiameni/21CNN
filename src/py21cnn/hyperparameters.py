from tensorflow import keras
def leakyrelu(x):
    return keras.activations.relu(x, alpha=0.1)

def HP():
    HyP = {}
    HyP["Loss"] = [[None, "mse"]]
    HyP["BatchSize"] = [20]
    HyP["LearningRate"] = [0.001, 0.0001]
    HyP["Dropout"] = [0.2]
    HyP["ReducingLR"] = [True]
    HyP["BatchNormalization"] = [True, False]
    HyP["Optimizer"] = [
                        [keras.optimizers.Adam, "Adam", {}],
                        [keras.optimizers.Nadam, "Nadam", {}],
                        ]
    HyP["ActivationFunction"] = [
                                ["relu", {"activation": keras.activations.relu, "kernel_initializer": keras.initializers.he_uniform()}],
                                ["leakyrelu", {"activation": leakyrelu, "kernel_initializer": keras.initializers.he_uniform()}],
                                ]
    return HyP

def HP_large():
    HyP = {}
    HyP["Loss"] = [[None, "mse"]]
    HyP["BatchSize"] = [20, 100]
    HyP["LearningRate"] = [0.01, 0.001, 0.0001]
    HyP["Dropout"] = [0.2, 0.5]
    HyP["ReducingLR"] = [True]
    HyP["BatchNormalization"] = [True, False]
    HyP["Optimizer"] = [
                        [keras.optimizers.RMSprop, "RMSprop", {}],
                        [keras.optimizers.SGD, "SGD", {}],
                        [keras.optimizers.SGD, "Momentum", {"momentum":0.9, "nesterov":True}],
                        # [keras.optimizers.Adadelta, "Adadelta", {}],
                        # [keras.optimizers.Adagrad, "Adagrad", {}],
                        [keras.optimizers.Adam, "Adam", {}],
                        # [keras.optimizers.Adam, "Adam", {"amsgrad":True}],
                        [keras.optimizers.Adamax, "Adamax", {}],
                        [keras.optimizers.Nadam, "Nadam", {}],
                        ]
    HyP["ActivationFunction"] = [
                                ["relu", {"activation": keras.activations.relu, "kernel_initializer": keras.initializers.he_uniform()}],
                                # [keras.layers.LeakyReLU(alpha=0.1), "leakyrelu"],
                                ["leakyrelu", {"activation": leakyrelu, "kernel_initializer": keras.initializers.he_uniform()}],
                                ["elu", {"activation": keras.activations.elu, "kernel_initializer": keras.initializers.he_uniform()}],
                                ["selu", {"activation": keras.activations.selu, "kernel_initializer": keras.initializers.lecun_normal()}],
                                # [keras.activations.exponential, "exponential"],
                                # [keras.activations.tanh, "tanh"],
                                ]
    return HyP