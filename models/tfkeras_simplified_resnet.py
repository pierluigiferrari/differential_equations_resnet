'''
A simplified ResNet architecture in which each ResNet block has only one convolutional
layer.

Copyright (C) 2019 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import tensorflow as tf
import numpy as np

from layers.tfkeras_layer_Conv2DAntisymmetric import Conv2DAntisymmetric

def single_layer_identity_block(input_tensor,
                                num_filters,
                                antisymmetric,
                                use_batch_norm,
                                stage,
                                block):
    '''
    Create a simplified ResNet identity block that consists of just a single convolutional layer.
    The identity block is used when the input tensor and the output tensor of the block have the
    same number of channels, in which case the main branch and the shortcut branch of the block can
    simply be added directly, because the output tensors of the two branches have the same dimensions.

    Arguments:
        input_tensor (4D tensor): The 4D input tensor of the shape (batch, heigh, width, channels).
        num_filters (int): The number of filters to be used for the convolution kernel.
        antisymmetric (bool): If `True`, the convolution matrix for this block will be antisymmetric,
            which is equivalent to the convolution kernel being skew-centrosymmetric. If `False`,
            the block will contain a regular convolutional layer instead.
        use_batch_norm (bool): If `True`, the convolution layer will be followed by a batch normalization
            layer.
        stage (int): The number of the current stage. Used for the generation of the layer names.
            Usually, a new stage begins after every pooling layer, i.e. whenever the spatial
            dimensions (height and width) of the convolutional feature map change.
        block (str): The label of the current block within the current stage. Used for
            the generation of the layer names. Usually, ResNet blocks within a stage will be labeled
            'a', 'b', 'c', etc.

    Returns:
        The output tensor for the block.
    '''

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if antisymmetric:
        x = Conv2DAntisymmetric(num_filters=num_filters,
                                trainable=True,
                                use_bias=True,
                                name=conv_name_base + '2')(input_tensor)
    else:
        x = tf.keras.layers.Conv2D(filters=num_filters,
                                   kernel_size=(3,3),
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   name=conv_name_base + '2')(input_tensor)

    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(axis=3,
                                               name=bn_name_base + '2')(x)

    x = tf.keras.layers.add([x, input_tensor])
    x = tf.keras.layers.Activation('relu')(x)

    return x

def single_layer_conv_block(input_tensor,
                            num_filters,
                            antisymmetric,
                            use_batch_norm,
                            stage,
                            block):
    '''
    Create a simplified ResNet convolutional block that consists of just a single convolutional layer.
    The convolutional block is used when the input tensor and the output tensor of the block do not
    have the same number of channels, in which case the number of channels of the shortcut branch gets
    increased or reduced to the number of channels of the main branch by means of a 1-by-1 convolution,
    so that the output tensors of the two branches have the same dimensions. Afterwards the output
    tensors of the two branches are added.

    Arguments:
        input_tensor (4D tensor): The 4D input tensor of the shape (batch, heigh, width, channels).
        num_filters (int): The number of filters to be used for the convolution kernel.
        antisymmetric (bool): If `True`, the convolution matrix for this block will be antisymmetric,
            which is equivalent to the convolution kernel being skew-centrosymmetric. If `False`,
            the block will contain a regular convolutional layer instead.
        use_batch_norm (bool): If `True`, the convolution layer will be followed by a batch normalization
            layer.
        stage (int): The number of the current stage. Used for the generation of the layer names.
            Usually, a new stage begins after every pooling layer, i.e. whenever the spatial
            dimensions (height and width) of the convolutional feature map change.
        block (str): The label of the current block within the current stage. Used for
            the generation of the layer names. Usually, ResNet blocks within a stage will be labeled
            'a', 'b', 'c', etc.

    Returns:
        The output tensor for the block.
    '''

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if antisymmetric:
        x = Conv2DAntisymmetric(num_filters=num_filters,
                                trainable=True,
                                use_bias=True,
                                name=conv_name_base + '2')(input_tensor)
    else:
        x = tf.keras.layers.Conv2D(filters=num_filters,
                                   kernel_size=(3,3),
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   name=conv_name_base + '2')(input_tensor)

    shortcut = tf.keras.layers.Conv2D(filters=num_filters,
                                      kernel_size=(1,1),
                                      kernel_initializer='he_normal',
                                      name=conv_name_base + '1')(input_tensor)

    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(axis=3,
                                               name=bn_name_base + '2')(x)

        shortcut = tf.keras.layers.BatchNormalization(axis=3,
                                                      name=bn_name_base + '1')(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x

def build_simplified_resnet(image_size,
                            num_classes,
                            architecture='antisymmetric',
                            use_batch_norm=False,
                            use_max_pooling=False,
                            subtract_mean=None,
                            divide_by_stddev=None):
    '''
    Build a simplified ResNet for image classification in which each ResNet block
    has only one convolutional layer.

    Arguments:
        num_classes (int): The number of classes for classification.
        architecture (str, optional): If 'antisymmetric', will build a ResNet with
            anti-centrosymmetric convolution kernels.
        use_batch_norm (bool, optional): If `True`, adds a batch normalization layer
            after each convolutional layer.
        use_max_pooling (bool, optional): If `True`, adds max pooling after certain
            blocks. If `False`, no pooling is performed.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
    '''

    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    if architecture == 'antisymmetric':
        antisymmetric = True
        name = 'simplified_resnet_antisymmetric'
    else:
        antisymmetric = False
        name = 'simplified_resnet_regular'

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_shift(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    ############################################################################
    # Build the network.
    ############################################################################

    input_tensor = tf.keras.layers.Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = tf.keras.layers.Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(input_tensor)
    if not (subtract_mean is None):
        x1 = tf.keras.layers.Lambda(input_mean_shift, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = tf.keras.layers.Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)

    # Stage 1
    x = single_layer_conv_block(x1, 16, antisymmetric, use_batch_norm, stage=1, block='a')
    x = single_layer_identity_block(x, 16, antisymmetric, use_batch_norm, stage=1, block='b')
    x = single_layer_conv_block(x, 24, antisymmetric, use_batch_norm, stage=1, block='c')
    x = single_layer_identity_block(x, 24, antisymmetric, use_batch_norm, stage=1, block='d')

    if use_max_pooling:
        # Stage 1 Pooling
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, name='stage1_pooling')(x)

    # Stage 2
    x = single_layer_conv_block(x, 32, antisymmetric, use_batch_norm, stage=2, block='a')
    x = single_layer_identity_block(x, 32, antisymmetric, use_batch_norm, stage=2, block='b')
    x = single_layer_conv_block(x, 24, antisymmetric, use_batch_norm, stage=2, block='c')
    x = single_layer_identity_block(x, 24, antisymmetric, use_batch_norm, stage=2, block='d')

    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax', name='fc')(x)

    # Create the model.
    model = tf.keras.models.Model(input_tensor, x, name=name)

    return model
