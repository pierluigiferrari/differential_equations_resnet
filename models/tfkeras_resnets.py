'''
Contains functions to build various ResNet architectures.

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

from tf.keras.layers import Conv2D, BatchNormalization, add, Activation, Input, Lambda, MaxPooling2D, GlobalAveragePooling2D, Dense, ZeroPadding2D

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
        x = Conv2D(filters=num_filters,
                   kernel_size=(3,3),
                   padding='same',
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2')(input_tensor)

    if use_batch_norm:
        x = BatchNormalization(axis=3,
                               name=bn_name_base + '2')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)

    return x

def bottleneck_identity_block(input_tensor,
                              num_filters,
                              antisymmetric,
                              use_batch_norm,
                              stage,
                              block):
    '''
    Create a regular ResNet bottleneck identity block that consists of three consecutive convolutional
    layers (1-by-1, 3-by-3, 1-by-1) as described in (https://arxiv.org/pdf/1512.03385.pdf).
    The identity block is used when the input tensor and the output tensor of the block have the
    same number of channels, in which case the main branch and the shortcut branch of the block can
    simply be added directly, because the output tensors of the two branches have the same dimensions.

    Arguments:
        input_tensor (4D tensor): The 4D input tensor of the shape (batch, heigh, width, channels).
        num_filters (tuple): A tuple of 3 integers which define the number of filters to be used
            for the three convolutional layers.
        antisymmetric (bool): If `True`, the convolution matrix of the 3-by-3 convolutional layer of
            this block will be antisymmetric, which is equivalent to the convolution kernel being
            skew-centrosymmetric. If `False`, the block will contain a regular 3-by-3 convolutional
            layer instead.
        use_batch_norm (bool): If `True`, each convolutional layer of this block will be followed by
            a batch normalization layer.
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

    ############################################################################
    #                                1-by-1
    ############################################################################

    x = Conv2D(filters=num_filters[0],
               kernel_size=(1,1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2a')(input_tensor)
    if use_batch_norm:
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    ############################################################################
    #                                3-by-3
    ############################################################################

    if antisymmetric:
        x = Conv2DAntisymmetric(num_filters=num_filters[1],
                                trainable=True,
                                use_bias=True,
                                name=conv_name_base + '2')(x)
    else:
        x = Conv2D(filters=num_filters[1],
                   kernel_size=(3,3),
                   padding='same',
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2b')(x)
    if use_batch_norm:
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    ############################################################################
    #                                1-by-1
    ############################################################################

    x = Conv2D(filters=num_filters[2],
               kernel_size=(1,1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2c')(x)
    if use_batch_norm:
        BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    ############################################################################
    #                                fusion
    ############################################################################

    x = add([x, input_tensor])
    x = Activation('relu')(x)

    return x

def single_layer_conv_block(input_tensor,
                            num_filters,
                            antisymmetric,
                            use_batch_norm,
                            stage,
                            block):
    '''
    Create a simplified ResNet conv block that consists of just a single convolutional layer.
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
        x = Conv2D(filters=num_filters,
                   kernel_size=(3,3),
                   padding='same',
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2')(input_tensor)

    shortcut = Conv2D(filters=num_filters,
                      kernel_size=(1,1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '1')(input_tensor)

    if use_batch_norm:
        x = BatchNormalization(axis=3,
                               name=bn_name_base + '2')(x)

        shortcut = BatchNormalization(axis=3,
                                      name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x

def bottleneck_conv_block(input_tensor,
                          num_filters,
                          antisymmetric,
                          use_batch_norm,
                          stage,
                          block,
                          strides=(1,1)):
    '''
    Create a regular ResNet bottleneck conv block that consists of three consecutive convolutional
    layers (1-by-1, 3-by-3, 1-by-1) as described in (https://arxiv.org/pdf/1512.03385.pdf).
    The convolutional block is used when the input tensor and the output tensor of the block do not
    have the same number of channels, in which case the number of channels of the shortcut branch gets
    increased or reduced to the number of channels of the main branch by means of a 1-by-1 convolution,
    so that the output tensors of the two branches have the same dimensions. Afterwards the output
    tensors of the two branches are added.

    Arguments:
        input_tensor (4D tensor): The 4D input tensor of the shape (batch, heigh, width, channels).
        num_filters (tuple): A tuple of 3 integers which define the number of filters to be used
            for the three convolutional layers.
        antisymmetric (bool): If `True`, the convolution matrix of the 3-by-3 convolutional layer of
            this block will be antisymmetric, which is equivalent to the convolution kernel being
            skew-centrosymmetric. If `False`, the block will contain a regular 3-by-3 convolutional
            layer instead.
        use_batch_norm (bool): If `True`, each convolutional layer of this block will be followed by
            a batch normalization layer.
        stage (int): The number of the current stage. Used for the generation of the layer names.
            Usually, a new stage begins after every pooling layer, i.e. whenever the spatial
            dimensions (height and width) of the convolutional feature map change.
        block (str): The label of the current block within the current stage. Used for
            the generation of the layer names. Usually, ResNet blocks within a stage will be labeled
            'a', 'b', 'c', etc.
        strides (tuple, optional): The spatial strides for the first 1-by-1 convolutional layer
            of this block.

    Returns:
        The output tensor for the block.
    '''

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    ############################################################################
    #                                1-by-1
    ############################################################################

    x = Conv2D(filters=num_filters[0],
               kernel_size=(1,1),
               strides=strides,
               kernel_initializer='he_normal',
               name=conv_name_base + '2a')(input_tensor)
    if use_batch_norm:
        x = BatchNormalization(axis=3,
                               name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    ############################################################################
    #                                3-by-3
    ############################################################################

    if antisymmetric:
        x = Conv2DAntisymmetric(num_filters=num_filters[1],
                                trainable=True,
                                use_bias=True,
                                name=conv_name_base + '2')(x)
    else:
        x = Conv2D(filters=num_filters[1],
                   kernel_size=(3,3),
                   padding='same',
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2b')(x)
    if use_batch_norm:
        x = BatchNormalization(axis=3,
                               name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    ############################################################################
    #                                1-by-1
    ############################################################################

    x = Conv2D(filters=num_filters[2],
               kernel_size=(1,1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2c')(x)
    if use_batch_norm:
        BatchNormalization(axis=3,
                           name=bn_name_base + '2c')(x)

    ############################################################################
    #                                shortcut
    ############################################################################

    shortcut = Conv2D(filters=num_filters[2],
                      kernel_size=(1,1),
                      strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '1')(input_tensor)
    if use_batch_norm:
        shortcut = BatchNormalization(axis=3,
                                      name=bn_name_base + '1')(shortcut)

    ############################################################################
    #                                fusion
    ############################################################################

    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x

def build_simplified_resnet(image_size,
                            num_classes,
                            architecture='antisymmetric',
                            use_batch_norm=False,
                            use_max_pooling=False,
                            subtract_mean=None,
                            divide_by_stddev=None,
                            include_top=True):
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
        include_top (bool, optional): If `False`, the output of the last convolutional layer is the model output.
            Otherwise, an average pooling layer and a fully connected layer followed by a softmax activation
            layer will be added, the output of the latter of which will be the model output.
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

    input_tensor = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(input_tensor)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_shift, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)

    # Stage 1
    x = single_layer_conv_block(x1, 16, antisymmetric, use_batch_norm, stage=1, block='a')
    x = single_layer_identity_block(x, 16, antisymmetric, use_batch_norm, stage=1, block='b')
    x = single_layer_conv_block(x, 24, antisymmetric, use_batch_norm, stage=1, block='c')
    x = single_layer_identity_block(x, 24, antisymmetric, use_batch_norm, stage=1, block='d')

    if use_max_pooling:
        # Stage 1 Pooling
        x = MaxPooling2D(pool_size=(2, 2), strides=None, name='stage1_pooling')(x)

    # Stage 2
    x = single_layer_conv_block(x, 32, antisymmetric, use_batch_norm, stage=2, block='a')
    x = single_layer_identity_block(x, 32, antisymmetric, use_batch_norm, stage=2, block='b')
    x = single_layer_conv_block(x, 24, antisymmetric, use_batch_norm, stage=2, block='c')
    x = single_layer_identity_block(x, 24, antisymmetric, use_batch_norm, stage=2, block='d')

    if include_top:
        x = GlobalAveragePooling2D(name='global_average_pooling')(x)
        x = Dense(num_classes, activation='softmax', name='fc')(x)

    # Create the model.
    model = tf.keras.models.Model(input_tensor, x, name=name)

    return model

def build_single_block_resnet(image_size,
                              architecture='antisymmetric',
                              blocks_per_stage=[3, 4, 6, 3],
                              filters_per_block=[64, 128, 256, 512],
                              num_classes=None,
                              use_batch_norm=False,
                              use_max_pooling=False,
                              subtract_mean=None,
                              divide_by_stddev=None,
                              include_top=True):
    '''
    Build a ResNet with in which each ResNet block has only one convolutional layer. The overall ResNet is composed
    of five stages of ResNet blocks. Each stage consists of one or more identical blocks.

    Arguments:
        image_size (tuple): A tuple `(height, width, channels)` of three integers representing the size
            and number of channels of the image input.
        architecture (str, optional): If 'antisymmetric', will build a ResNet in which
            all 3-by-3 convolution kernels are anti-centrosymmetric.
        blocks_per_stage (tuple, optional): A tuple of four positive integers representing the number of
            ResNet blocks for the stages 2, 3, 4, and 5 of the ResNet.
        filters_per_block (tuple, optional): A tuple of four positive integers representing the number of
            filters to be used for the convolutional layers of the blocks in each of the stages 2, 3, 4, and 5
            of the ResNet.
        num_classes (int, optional): The number of classes for classification. Only relevant if `inclue_top`
            is `True`.
        use_batch_norm (bool, optional): If `True`, adds a batch normalization layer
            after each convolutional layer.
        use_max_pooling (bool, optional): If `True`, adds max pooling after the stages 1, 2, 3, and 4.
            If `False`, no pooling is performed apart from the global average pooling at the end of the network.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        include_top (bool, optional): If `False`, the output of the last convolutional layer is the model output.
            Otherwise, an average pooling layer and a fully connected layer with `num_classes` outputs followed
            by a softmax activation ayer will be added, the output of the latter of which will be the model output.
    '''

    if include_top and (num_classes is None):
        raise ValueError("You must pass a positive integer for `num_classes` if `include_top` is `True`.")

    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    name = 'single_block_resnet'

    if architecture == 'antisymmetric':
        antisymmetric = True
        name += '_antisymmetric'
    else:
        antisymmetric = False
        name += '_regular'

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

    input_tensor = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(input_tensor)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_shift, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)

    # Stage 1
    if antisymmetric:
        x = Conv2DAntisymmetric(num_filters=64,
                                trainable=True,
                                use_bias=True,
                                strides=(2,2),
                                name='conv1')(x1)
    else:
        x = Conv2D(filters=64,
                   kernel_size=(3,3),
                   strides=(2,2),
                   padding='same',
                   kernel_initializer='he_normal',
                   name='conv1')(x1)
    if use_batch_norm:
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    if use_max_pooling:
        x = ZeroPadding2D(padding=(1,1), name='pool1_pad')(x)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='stage1_pooling')(x)

    # Stage 2
    x = single_layer_conv_block(x, filters_per_block[0], antisymmetric, use_batch_norm, stage=2, block='a')
    for i in range(ord('b'), ord('b') + blocks_per_stage[0] - 1):
        x = single_layer_identity_block(x, filters_per_block[0], antisymmetric, use_batch_norm, stage=2, block=chr(i))
    if use_max_pooling:
        x = MaxPooling2D(pool_size=(2, 2), strides=None, name='stage2_pooling')(x)

    # Stage 3
    x = single_layer_conv_block(x, filters_per_block[1], antisymmetric, use_batch_norm, stage=3, block='a')
    for i in range(ord('b'), ord('b') + blocks_per_stage[1] - 1):
        x = single_layer_identity_block(x, filters_per_block[1], antisymmetric, use_batch_norm, stage=3, block=chr(i))
    if use_max_pooling:
        x = MaxPooling2D(pool_size=(2, 2), strides=None, name='stage3_pooling')(x)

    # Stage 4
    x = single_layer_conv_block(x, filters_per_block[2], antisymmetric, use_batch_norm, stage=4, block='a')
    for i in range(ord('b'), ord('b') + blocks_per_stage[2] - 1):
        x = single_layer_identity_block(x, filters_per_block[2], antisymmetric, use_batch_norm, stage=4, block=chr(i))
    if use_max_pooling:
        x = MaxPooling2D(pool_size=(2, 2), strides=None, name='stage4_pooling')(x)

    # Stage 5
    x = single_layer_conv_block(x, filters_per_block[3], antisymmetric, use_batch_norm, stage=5, block='a')
    for i in range(ord('b'), ord('b') + blocks_per_stage[3] - 1):
        x = single_layer_identity_block(x, filters_per_block[3], antisymmetric, use_batch_norm, stage=5, block=chr(i))

    if include_top:
        x = GlobalAveragePooling2D(name='global_average_pooling')(x)
        x = Dense(num_classes, activation='softmax', name='fc')(x)

    # Create the model.
    model = tf.keras.models.Model(input_tensor, x, name=name)

    return model

def build_resnet(image_size,
                 architecture='antisymmetric',
                 blocks_per_stage=[3, 4, 6, 3],
                 filters_per_block=[[64, 64, 256],
                                    [128, 128, 512],
                                    [256, 256, 1024],
                                    [512, 512, 2048]],
                 num_classes=None,
                 use_batch_norm=False,
                 subtract_mean=None,
                 divide_by_stddev=None,
                 include_top=True):
    '''
    Build a ResNet with the regular bottleneck block architecture. The overall ResNet is composed
    of five stages of ResNet blocks. Each stage consists of one or more identical blocks.

    Arguments:
        image_size (tuple): A tuple `(height, width, channels)` of three integers representing the size
            and number of channels of the image input.
        architecture (str, optional): If 'antisymmetric', will build a ResNet in which
            all 3-by-3 convolution kernels are anti-centrosymmetric.
        blocks_per_stage (tuple, optional): A tuple of four positive integers representing the number of
            ResNet blocks for the stages 2, 3, 4, and 5 of the ResNet. The default configuration produces
            a ResNet50. To produce a ResNet101 or ResNet152, use the configurations `[3, 4, 23, 3]` and
            `[3, 8, 36, 3]`, respectively.
        filters_per_block (tuple, optional): A tuple of four 3-tuples. The four tuples represent the number of
            filters to be used for the convolutional layers of the blocks in each of the stages 2, 3, 4, and 5
            of the ResNet. Each 3-tuple defines the number of filters to be used for the three convolutional
            layers (1-by-1, 3-by-3, 1-by-1) of the blocks in the respective stage.
        num_classes (int, optional): The number of classes for classification. Only relevant if `inclue_top`
            is `True`.
        use_batch_norm (bool, optional): If `True`, adds a batch normalization layer
            after each convolutional layer.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        include_top (bool, optional): If `False`, the output of the last convolutional layer is the model output.
            Otherwise, an average pooling layer and a fully connected layer with `num_classes` outputs followed
            by a softmax activation ayer will be added, the output of the latter of which will be the model output.
    '''

    if include_top and (num_classes is None):
        raise ValueError("You must pass a positive integer for `num_classes` if `include_top` is `True`.")

    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    name = 'resnet'

    if architecture == 'antisymmetric':
        antisymmetric = True
        name += '_antisymmetric'
    else:
        antisymmetric = False
        name += '_regular'

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

    input_tensor = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(input_tensor)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_shift, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)

    # Stage 1
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x1)
    x = Conv2D(filters=64,
               kernel_size=(7, 7),
               strides=(2, 2),
               padding='valid',
               kernel_initializer='he_normal',
               name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='stage1_pooling')(x)

    # Stage 2
    x = bottleneck_conv_block(x, filters_per_block[0], antisymmetric, use_batch_norm, stage=2, block='a', strides=(1,1))
    for i in range(ord('b'), ord('b') + blocks_per_stage[0] - 1):
        x = bottleneck_identity_block(x, filters_per_block[0], antisymmetric, use_batch_norm, stage=2, block=chr(i))

    # Stage 3
    x = bottleneck_conv_block(x, filters_per_block[1], antisymmetric, use_batch_norm, stage=3, block='a', strides=(2,2))
    for i in range(ord('b'), ord('b') + blocks_per_stage[1] - 1):
        x = bottleneck_identity_block(x, filters_per_block[1], antisymmetric, use_batch_norm, stage=3, block=chr(i))

    # Stage 4
    x = bottleneck_conv_block(x, filters_per_block[2], antisymmetric, use_batch_norm, stage=4, block='a', strides=(2,2))
    for i in range(ord('b'), ord('b') + blocks_per_stage[2] - 1):
        x = bottleneck_identity_block(x, filters_per_block[2], antisymmetric, use_batch_norm, stage=4, block=chr(i))

    # Stage 5
    x = bottleneck_conv_block(x, filters_per_block[3], antisymmetric, use_batch_norm, stage=5, block='a', strides=(2,2))
    for i in range(ord('b'), ord('b') + blocks_per_stage[3] - 1):
        x = bottleneck_identity_block(x, filters_per_block[3], antisymmetric, use_batch_norm, stage=5, block=chr(i))

    if include_top:
        x = GlobalAveragePooling2D(name='global_average_pooling')(x)
        x = Dense(num_classes, activation='softmax', name='fc')(x)

    # Create the model.
    model = tf.keras.models.Model(input_tensor, x, name=name)

    return model
