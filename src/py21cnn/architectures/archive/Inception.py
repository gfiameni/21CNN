from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow import keras
backend = keras.backend
layers = keras.layers
models = keras.models
keras_utils = keras.utils

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None,
              ):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x

def conv2d( x,
            filters,
            num_row,
            num_col,
            padding='same',
            strides=(1, 1),
            name=None,
            ):
    if name is not None:
        conv_name = name + '_conv'
    else:
        conv_name = None
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    return x

def V3(dense_layers_pow2 = range(9, 2, -1),
                dropout = 0.2,
                # weights='imagenet',
                # input_tensor=None,
                input_shape=None,
                bn = False,
                ):
    """Instantiates the Inception v3 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils

    if bn:
        conv2D = conv2d_bn
    else:
        conv2D = conv2d
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    # if not (weights in {'imagenet', None} or os.path.exists(weights)):
    #     raise ValueError('The `weights` argument should be either '
    #                      '`None` (random initialization), `imagenet` '
    #                      '(pre-training on ImageNet), '
    #                      'or the path to the weights file to be loaded.')

    # if weights == 'imagenet' and include_top and classes != 1000:
    #     raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
    #                      ' as true, `classes` should be 1000')

    # Determine proper input shape
    # input_shape = _obtain_input_shape(
    #     input_shape,
    #     default_size=299,
    #     min_size=75,
    #     data_format=backend.image_data_format(),
    #     require_flatten=include_top,
    #     weights=weights)

    # input_shape = input_shape

    # if input_tensor is None:
    #     img_input = layers.Input(shape=input_shape)
    # else:
    #     if not backend.is_keras_tensor(input_tensor):
    #         img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor

    img_input = layers.Input(shape=input_shape)

    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # x = conv2D(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2D(img_input, 32, 3, 3, padding='valid')
    x = conv2D(x, 32, 3, 3, padding='valid')
    x = conv2D(x, 64, 3, 3)
    # x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2D(x, 80, 1, 1, padding='valid')
    x = conv2D(x, 192, 3, 3, padding='valid')
    # x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)


    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2D(x, 64, 1, 1)

    branch5x5 = conv2D(x, 48, 1, 1)
    branch5x5 = conv2D(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2D(x, 64, 1, 1)
    branch3x3dbl = conv2D(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2D(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2D(x, 64, 1, 1)

    branch5x5 = conv2D(x, 48, 1, 1)
    branch5x5 = conv2D(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2D(x, 64, 1, 1)
    branch3x3dbl = conv2D(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2D(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2D(x, 64, 1, 1)

    branch5x5 = conv2D(x, 48, 1, 1)
    branch5x5 = conv2D(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2D(x, 64, 1, 1)
    branch3x3dbl = conv2D(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2D(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2D(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2D(x, 64, 1, 1)
    branch3x3dbl = conv2D(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2D(x, 192, 1, 1)

    branch7x7 = conv2D(x, 128, 1, 1)
    branch7x7 = conv2D(branch7x7, 128, 1, 7)
    branch7x7 = conv2D(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2D(x, 128, 1, 1)
    branch7x7dbl = conv2D(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2D(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2D(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2D(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2D(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2D(x, 192, 1, 1)

        branch7x7 = conv2D(x, 160, 1, 1)
        branch7x7 = conv2D(branch7x7, 160, 1, 7)
        branch7x7 = conv2D(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2D(x, 160, 1, 1)
        branch7x7dbl = conv2D(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2D(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2D(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2D(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2D(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2D(x, 192, 1, 1)

    branch7x7 = conv2D(x, 192, 1, 1)
    branch7x7 = conv2D(branch7x7, 192, 1, 7)
    branch7x7 = conv2D(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2D(x, 192, 1, 1)
    branch7x7dbl = conv2D(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2D(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2D(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2D(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2D(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2D(x, 192, 1, 1)
    branch3x3 = conv2D(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2D(x, 192, 1, 1)
    branch7x7x3 = conv2D(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2D(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2D(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2D(x, 320, 1, 1)

        branch3x3 = conv2D(x, 384, 1, 1)
        branch3x3_1 = conv2D(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2D(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2D(x, 448, 1, 1)
        branch3x3dbl = conv2D(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2D(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2D(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2D(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))


    # x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = conv2D(x, 64, 4, 4, padding='valid')
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(2**11, activation='relu', name=f'dense{2**11}')(x)
    x = layers.Dense(2**10, activation='relu', name=f'dense{2**10}')(x)

    for i in dense_layers_pow2:
        x = layers.Dense(2**i, activation='relu', name=f'dense{2**i}')(x)

    x = layers.Dense(4, activation='linear', name = 'out')(x)


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    # if input_tensor is not None:
    #     inputs = keras_utils.get_source_inputs(input_tensor)
    # else:
    #     inputs = img_input
    inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='inception_v3')

    # # Load weights.
    # if weights == 'imagenet':
    #     if include_top:
    #         weights_path = keras_utils.get_file(
    #             'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
    #             WEIGHTS_PATH,
    #             cache_subdir='models',
    #             file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
    #     else:
    #         weights_path = keras_utils.get_file(
    #             'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #             WEIGHTS_PATH_NO_TOP,
    #             cache_subdir='models',
    #             file_hash='bcbd6486424b2319ff4ef7d526e38f63')
    #     model.load_weights(weights_path)
    # elif weights is not None:
    #     model.load_weights(weights)

    return model


def V3_reduced(dense_layers_pow2 = range(8, 2, -1),
                dropout = 0.2,
                # weights='imagenet',
                # input_tensor=None,
                input_shape=None,
                bn=False,
                ):

    global backend, layers, models, keras_utils

    if bn:
        conv2D = conv2d_bn
    else:
        conv2D = conv2d

    img_input = layers.Input(shape=input_shape)

    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2D(img_input, 32, 3, 3, padding='valid')
    # x = conv2D(x, 32, 3, 3, padding='valid')
    x = conv2D(x, 64, 3, 3)
    x = conv2D(x, 80, 1, 1, padding='valid')
    x = conv2D(x, 192, 3, 3, padding='valid')
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)


    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2D(x, 64, 1, 1)

    branch5x5 = conv2D(x, 48, 1, 1)
    branch5x5 = conv2D(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2D(x, 64, 1, 1)
    branch3x3dbl = conv2D(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2D(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2D(x, 64, 1, 1)

    branch5x5 = conv2D(x, 48, 1, 1)
    branch5x5 = conv2D(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2D(x, 64, 1, 1)
    branch3x3dbl = conv2D(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2D(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # # mixed 2: 35 x 35 x 288
    # branch1x1 = conv2D(x, 64, 1, 1)

    # branch5x5 = conv2D(x, 48, 1, 1)
    # branch5x5 = conv2D(branch5x5, 64, 5, 5)

    # branch3x3dbl = conv2D(x, 64, 1, 1)
    # branch3x3dbl = conv2D(branch3x3dbl, 96, 3, 3)
    # branch3x3dbl = conv2D(branch3x3dbl, 96, 3, 3)

    # branch_pool = layers.AveragePooling2D((3, 3),
    #                                       strides=(1, 1),
    #                                       padding='same')(x)
    # branch_pool = conv2D(branch_pool, 64, 1, 1)
    # x = layers.concatenate(
    #     [branch1x1, branch5x5, branch3x3dbl, branch_pool],
    #     axis=channel_axis,
    #     name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2D(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2D(x, 64, 1, 1)
    branch3x3dbl = conv2D(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2D(x, 192, 1, 1)

    branch7x7 = conv2D(x, 128, 1, 1)
    branch7x7 = conv2D(branch7x7, 128, 1, 7)
    branch7x7 = conv2D(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2D(x, 128, 1, 1)
    branch7x7dbl = conv2D(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2D(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2D(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2D(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2D(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # # mixed 5, 6: 17 x 17 x 768
    # for i in range(2):
    #     branch1x1 = conv2D(x, 192, 1, 1)

    #     branch7x7 = conv2D(x, 160, 1, 1)
    #     branch7x7 = conv2D(branch7x7, 160, 1, 7)
    #     branch7x7 = conv2D(branch7x7, 192, 7, 1)

    #     branch7x7dbl = conv2D(x, 160, 1, 1)
    #     branch7x7dbl = conv2D(branch7x7dbl, 160, 7, 1)
    #     branch7x7dbl = conv2D(branch7x7dbl, 160, 1, 7)
    #     branch7x7dbl = conv2D(branch7x7dbl, 160, 7, 1)
    #     branch7x7dbl = conv2D(branch7x7dbl, 192, 1, 7)

    #     branch_pool = layers.AveragePooling2D(
    #         (3, 3), strides=(1, 1), padding='same')(x)
    #     branch_pool = conv2D(branch_pool, 192, 1, 1)
    #     x = layers.concatenate(
    #         [branch1x1, branch7x7, branch7x7dbl, branch_pool],
    #         axis=channel_axis,
    #         name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2D(x, 192, 1, 1)

    branch7x7 = conv2D(x, 192, 1, 1)
    branch7x7 = conv2D(branch7x7, 192, 1, 7)
    branch7x7 = conv2D(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2D(x, 192, 1, 1)
    branch7x7dbl = conv2D(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2D(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2D(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2D(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2D(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2D(x, 192, 1, 1)
    branch3x3 = conv2D(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2D(x, 192, 1, 1)
    branch7x7x3 = conv2D(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2D(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2D(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    # mixed 9: 8 x 8 x 2048
    # for i in range(2):
    for i in range(1):
        branch1x1 = conv2D(x, 320, 1, 1)

        branch3x3 = conv2D(x, 384, 1, 1)
        branch3x3_1 = conv2D(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2D(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2D(x, 448, 1, 1)
        branch3x3dbl = conv2D(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2D(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2D(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2D(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    # x = conv2D(x, 64, 4, 4, padding='valid')
    # x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(2**10, activation='relu', name=f'dense{2**10}')(x)

    for i in dense_layers_pow2:
        x = layers.Dense(2**i, activation='relu', name=f'dense{2**i}')(x)

    x = layers.Dense(4, activation='linear', name = 'out')(x)


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    # if input_tensor is not None:
    #     inputs = keras_utils.get_source_inputs(input_tensor)
    # else:
    #     inputs = img_input
    inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='inception_v3')

    # # Load weights.
    # if weights == 'imagenet':
    #     if include_top:
    #         weights_path = keras_utils.get_file(
    #             'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
    #             WEIGHTS_PATH,
    #             cache_subdir='models',
    #             file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
    #     else:
    #         weights_path = keras_utils.get_file(
    #             'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #             WEIGHTS_PATH_NO_TOP,
    #             cache_subdir='models',
    #             file_hash='bcbd6486424b2319ff4ef7d526e38f63')
    #     model.load_weights(weights_path)
    # elif weights is not None:
    #     model.load_weights(weights)

    return model