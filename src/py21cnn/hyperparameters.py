from tensorflow import keras
def leakyrelu(x):
    return keras.activations.relu(x, alpha=0.1)

def HP(inputs):
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