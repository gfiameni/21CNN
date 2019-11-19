import keras
backend = keras.backend
layers = keras.layers
models = keras.models
keras_utils = keras.utils

def Model(  input_shape,
            pow2max = 10,
            dropout = False,
            LeakyAlpha=0.05,
            ):

    img_input = layers.Input(shape=input_shape)
    x = layers.Flatten()(img_input)
    x = layers.Dense(2**pow2max, name=f'dense{2**pow2max}')(x)
    x = layers.LeakyReLU(alpha=LeakyAlpha)(x)
   
    if dropout:
        layers.Dropout(dropout)(x)

    for i in range(pow2max-2, 2, -2):
            x = layers.Dense(2**i, name=f'dense{2**i}')(x)
            x = layers.LeakyReLU(alpha=LeakyAlpha)(x)

    x = layers.Dense(4, name='out')(x)

    m = models.Model(img_input, x, name='DNN')
    m.summary()
    return m