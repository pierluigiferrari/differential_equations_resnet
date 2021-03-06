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

from tensorflow.keras.layers import Conv2D, BatchNormalization, add, Activation, Input, Lambda, MaxPooling2D, GlobalAveragePooling2D, Dense, ZeroPadding2D
from tensorflow.keras.regularizers import l2

from layers.tfkeras_layer_Conv2DAntisymmetric3By3 import Conv2DAntisymmetric3By3

def single_layer_identity_block(input_tensor,
                                kernel_size,
                                antisymmetric,
                                use_batch_norm,
                                stage,
                                block,
                                h=1.0,
                                gamma=0.0,
                                kernel_regularizer=None,
                                bias_regularizer=None):
    '''
    Create a simplified ResNet identity block that consists of just a single convolutional layer.
    The identity block is used when the input tensor and the output tensor of the block have the
    same number of channels, in which case the main branch and the shortcut branch of the block can
    simply be added directly, because the output tensors of the two branches have the same dimensions.

    Arguments:
        input_tensor (4D tensor): The 4D input tensor of the shape (batch, heigh, width, channels).
        kernel_size (int): The size of the quadratic convolution kernel.
        antisymmetric (bool): If `True`, the convolution matrix for this block will be antisymmetric,
            which is equivalent to the convolution kernel being skew-centrosymmetric. If `False`,
            the block will contain a regular convolutional layer instead.
        use_batch_norm (bool): If `True`, the convolution layer will be followed by a batch normalization
            layer.
        stage (int): The number of the current stage. Used for the generation of the layer names.
            Usually, a new stage begins whenever the spatial dimensions (height and width) of the
            convolutional feature map change.
        block (int): The number of the current block within the current stage. Used for
            the generation of the layer names.
        h (float, optional): TODO
        gamma (float, optional): TODO
        kernel_regularizer (tf.keras.regularizer, optional): An instance of `tf.keras.regularizer`.
        bias_regularizer (tf.keras.regularizer, optional): An instance of `tf.keras.regularizer`.

    Returns:
        The output tensor for the block.
    '''

    conv_name_base = 'res' + str(stage) + '_' + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + str(block) + '_branch'

    if antisymmetric:
        x = Conv2DAntisymmetric3By3(gamma=gamma,
                                    strides=(1, 1),
                                    use_bias=True,
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=kernel_regularizer,
                                    name=conv_name_base + '2')(input_tensor)
    else:
        x = Conv2D(filters=int(input_tensor.shape[-1]),
                   kernel_size=kernel_size,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   name=conv_name_base + '2')(input_tensor)

    if use_batch_norm:
        x = BatchNormalization(axis=3,
                               name=bn_name_base + '2')(x)

    x = Activation('relu')(x)
    if h != 1.0:
        x = Lambda(lambda x: h * x, name='scale' + str(stage) + '_' + str(block))(x)
    x = add([x, input_tensor])

    return x

def bottleneck_identity_block(input_tensor,
                              kernel_size,
                              num_filters,
                              antisymmetric,
                              use_batch_norm,
                              stage,
                              block,
                              gamma=0.0,
                              kernel_regularizer=None,
                              bias_regularizer=None):
    '''
    Create a regular ResNet bottleneck identity block that consists of three consecutive convolutional
    layers (1-by-1, k-by-k, 1-by-1) as described in (https://arxiv.org/pdf/1512.03385.pdf).
    The identity block is used when the input tensor and the output tensor of the block have the
    same number of channels, in which case the main branch and the shortcut branch of the block can
    simply be added directly, because the output tensors of the two branches have the same dimensions.

    Arguments:
        input_tensor (4D tensor): The 4D input tensor of the shape (batch, heigh, width, channels).
        kernel_size (int): The size of the quadratic convolution kernel.
        num_filters (tuple): A tuple of 3 integers which define the number of filters to be used
            for the three convolutional layers. The second element of the tuple can also be `None`,
            indicating that the number of filters of the block's second convolutional layer will be
            identical to its number of input channels. This is required in order to yield an antisymmetric
            convolution matrix for the second conv layer.
        antisymmetric (bool): If `True`, the convolution matrix of the second convolutional layer of
            this block will be antisymmetric, which is equivalent to the convolution kernel being
            skew-centrosymmetric. If `False`, the block will contain a regular 3-by-3 convolutional
            layer instead. Setting this argument to `True` only has an effect if the second element of
            `num_filters` is set to `None`, because the convolution matrix can only be a square matrix
            if the number of input channels to and output channels from the layer are identical.
        use_batch_norm (bool): If `True`, each convolutional layer of this block will be followed by
            a batch normalization layer.
        stage (int): The number of the current stage. Used for the generation of the layer names.
            Usually, a new stage begins whenever the spatial dimensions (height and width) of the
            convolutional feature map change.
        block (int): The number of the current block within the current stage. Used for
            the generation of the layer names.
        gamma (float, optional): TODO
        kernel_regularizer (tf.keras.regularizer, optional): An instance of `tf.keras.regularizer`.
        bias_regularizer (tf.keras.regularizer, optional): An instance of `tf.keras.regularizer`.

    Returns:
        The output tensor for the block.
    '''

    conv_name_base = 'res' + str(stage) + '_' + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + str(block) + '_branch'

    ############################################################################
    #                                1-by-1
    ############################################################################

    x = Conv2D(filters=num_filters[0],
               kernel_size=(1,1),
               kernel_initializer='he_normal',
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               name=conv_name_base + '2a')(input_tensor)
    if use_batch_norm:
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    ############################################################################
    #                                3-by-3
    ############################################################################

    if antisymmetric and (num_filters[1] is None):
        x = Conv2DAntisymmetric3By3(gamma=gamma,
                                    strides=(1, 1),
                                    use_bias=True,
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=kernel_regularizer,
                                    name=conv_name_base + '2b')(x)
    else:
        x = Conv2D(filters=num_filters[1],
                   kernel_size=kernel_size,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
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
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               name=conv_name_base + '2c')(x)
    if use_batch_norm:
        x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    ############################################################################
    #                                fusion
    ############################################################################

    x = add([x, input_tensor])
    x = Activation('relu')(x)

    return x

def single_layer_conv_block(input_tensor,
                            kernel_size,
                            num_filters,
                            strides,
                            use_batch_norm,
                            stage,
                            block,
                            kernel_regularizer=None,
                            bias_regularizer=None):
    '''
    Create a simplified ResNet conv block that consists of just a single convolutional layer.
    The convolutional block is used when the input tensor and the output tensor of the block do not
    have the same number of channels, in which case the number of channels of the shortcut branch gets
    increased or reduced to the number of channels of the main branch by means of a 1-by-1 convolution,
    so that the output tensors of the two branches have the same dimensions. Afterwards the output
    tensors of the two branches are added.

    Arguments:
        input_tensor (4D tensor): The 4D input tensor of the shape (batch, heigh, width, channels).
        kernel_size (int): The size of the quadratic convolution kernel.
        num_filters (int): The number of filters to be used for the convolution kernel.
        use_batch_norm (bool): If `True`, the convolution layer will be followed by a batch normalization
            layer.
        stage (int): The number of the current stage. Used for the generation of the layer names.
            Usually, a new stage begins whenever the spatial dimensions (height and width) of the
            convolutional feature map change.
        block (int): The number of the current block within the current stage. Used for
            the generation of the layer names.
        kernel_regularizer (tf.keras.regularizer, optional): An instance of `tf.keras.regularizer`.
        bias_regularizer (tf.keras.regularizer, optional): An instance of `tf.keras.regularizer`.

    Returns:
        The output tensor for the block.
    '''

    conv_name_base = 'res' + str(stage) + '_' + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + str(block) + '_branch'

    x = Conv2D(filters=num_filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               name=conv_name_base + '2')(input_tensor)

    shortcut = Conv2D(filters=num_filters,
                      kernel_size=(1,1),
                      strides=strides,
                      kernel_initializer='he_normal',
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer,
                      name=conv_name_base + '1')(input_tensor)

    if use_batch_norm:
        x = BatchNormalization(axis=3,
                               name=bn_name_base + '2')(x)

        shortcut = BatchNormalization(axis=3,
                                      name=bn_name_base + '1')(shortcut)

    x = Activation('relu')(x)
    x = add([x, shortcut])

    return x

def bottleneck_conv_block(input_tensor,
                          kernel_size,
                          num_filters,
                          antisymmetric,
                          use_batch_norm,
                          stage,
                          block,
                          version=1,
                          strides=(1,1),
                          gamma=0.0,
                          kernel_regularizer=None,
                          bias_regularizer=None):
    '''
    Create a regular ResNet bottleneck conv block that consists of three consecutive convolutional
    layers (1-by-1, k-by-k, 1-by-1) as described in (https://arxiv.org/pdf/1512.03385.pdf).
    The convolutional block is used when the input tensor and the output tensor of the block do not
    have the same number of channels, in which case the number of channels of the shortcut branch gets
    increased or reduced to the number of channels of the main branch by means of a 1-by-1 convolution,
    so that the output tensors of the two branches have the same dimensions. Afterwards the output
    tensors of the two branches are added.

    Arguments:
        input_tensor (4D tensor): The 4D input tensor of the shape (batch, heigh, width, channels).
        kernel_size (int): The size of the quadratic convolution kernel.
        num_filters (tuple): A tuple of 3 integers which define the number of filters to be used
            for the three convolutional layers. The second element of the tuple can also be `None`,
            indicating that the number of filters of the block's second convolutional layer will be
            identical to its number of input channels. This is required in order to yield an antisymmetric
            convolution matrix for the second conv layer.
        antisymmetric (bool): If `True`, the convolution matrix of the second convolutional layer of
            this block will be antisymmetric, which is equivalent to the convolution kernel being
            skew-centrosymmetric. If `False`, the block will contain a regular 3-by-3 convolutional
            layer instead. Setting this argument to `True` only has an effect if the second element of
            `num_filters` is set to `None`, because the convolution matrix can only be a square matrix
            if the number of input channels to and output channels from the layer are identical.
        use_batch_norm (bool): If `True`, each convolutional layer of this block will be followed by
            a batch normalization layer.
        stage (int): The number of the current stage. Used for the generation of the layer names.
            Usually, a new stage begins whenever the spatial dimensions (height and width) of the
            convolutional feature map change.
        block (int): The number of the current block within the current stage. Used for
            the generation of the layer names.
        version (float, optional): A value from the set {1, 1.5}, the version of the ResNet.
            The different versions are defined as follows:

            v1: The striding of the conv block is performed by the first 1-by-1 convolution.
                The order of operations of each convolutional layer is (conv, BN, ReLU).
            v1.5: The striding of the conv block is performed by the 3-by-3 convolution.
                The order of operations of each convolutional layer is (conv, BN, ReLU).
            v2 (currently not supported): The striding of the conv block is performed by
                the first 3-by-3 convolution. The order of operations of each convolutional layer
                is (BN, ReLU, conv).

            The reasoning for introducing v1.5 is that is has been reported that performing the
            striding (i.e. the spatial dimensionality reduction) in the 3-by-3 convolution instead
            of the 1-by-1 convolution results in higher and more stable accuracy in fewer epochs
            than the original v1 and has shown to scale to higher batch sizes with minimal degradation
            in accuracy. However, it has also been reported that v1.5 requires ~12% more compute to
            train and has 6% reduced throughput for inference compared to v1.
        strides (tuple, optional): The spatial strides for the first 1-by-1 convolutional layer
            (v1) or the 3-by-3 convolutional layer (v1.5) of this block.
        gamma (float, optional): TODO

    Returns:
        The output tensor for the block.
    '''

    if version == 1:
        strides_1_by_1 = strides
        strides_k_by_k = (1,1)
    elif version == 1.5:
        strides_1_by_1 = (1,1)
        strides_k_by_k = strides
    else:
        raise ValueError("Supported values for `version` are 1 and 1.5.")

    conv_name_base = 'res' + str(stage) + '_' + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + str(block) + '_branch'

    ############################################################################
    #                                1-by-1
    ############################################################################

    x = Conv2D(filters=num_filters[0],
               kernel_size=(1,1),
               strides=strides_1_by_1,
               kernel_initializer='he_normal',
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               name=conv_name_base + '2a')(input_tensor)
    if use_batch_norm:
        x = BatchNormalization(axis=3,
                               name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    ############################################################################
    #                                3-by-3
    ############################################################################

    if antisymmetric and (num_filters[1] is None):
        x = Conv2DAntisymmetric3By3(gamma=gamma,
                                    strides=strides_k_by_k,
                                    use_bias=True,
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=kernel_regularizer,
                                    name=conv_name_base + '2b')(x)
    else:
        x = Conv2D(filters=num_filters[1],
                   kernel_size=kernel_size,
                   strides=strides_k_by_k,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
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
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               name=conv_name_base + '2c')(x)
    if use_batch_norm:
        x = BatchNormalization(axis=3,
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

def build_single_block_resnet(image_shape,
                              kernel_type='antisymmetric',
                              kernel_size=3,
                              h=1.0,
                              gamma=0.0,
                              num_stages=5,
                              blocks_per_stage=[3, 4, 6, 3],
                              filters_per_block=[64, 128, 256, 512],
                              strides=[(2,2),(2,2),(2,2),(2,2)],
                              include_top=True,
                              fc_activation='softmax',
                              num_classes=None,
                              use_batch_norm=False,
                              use_max_pooling=[False, False, False, False],
                              l2_regularization=0.0,
                              subtract_mean=None,
                              divide_by_stddev=None,
                              verbose=False):
    '''
    Build a ResNet with in which each ResNet block has only one convolutional layer. The overall ResNet is composed
    of five stages of ResNet blocks. Each stage consists of one or more identical blocks.

    Arguments:
        image_shape (tuple): A tuple `(height, width, channels)` of three integers representing the size
            and number of channels of the image input.
        kernel_type (str, optional): If 'antisymmetric', will build a ResNet in which
            all 3-by-3 convolution kernels are anti-centrosymmetric.
        kernel_size (int, optional):
        h (float, optional): The scaling factor for the output of the residual connections. This scaling factor
            is 1.0 for the original ResNet architectures.
        gamma (float, optional): Only relevant if `kernel_type` is set to 'antisymmetric'. For an antisymmetric kernel,
            the real parts of the eigenvalues of the associated convolution matrix will be `gamma`.
        num_stages (int, optional):
        blocks_per_stage (tuple, optional): A tuple of four positive integers representing the number of
            ResNet blocks for the stages 2, 3, 4, and 5 of the ResNet.
        filters_per_block (tuple, optional): A tuple of four positive integers representing the number of
            filters to be used for the convolutional layers of the blocks in each of the stages 2, 3, 4, and 5
            of the ResNet.
        include_top (bool, optional): If `False`, the output of the last convolutional layer is the model output.
            Otherwise, an average pooling layer and a fully connected layer with `num_classes` outputs followed
            by a softmax activation ayer will be added, the output of the latter of which will be the model output.
        fc_activation (str, optional): The activation function to use for the very last layer of the network,
            i.e. the dense layer. Can be any valid Keras activation function name, e.g. 'softmax' for classification.
            If this is `None`, no activation will be applied to the dense layer. Only relevant if `include_top`
            is True.
        num_classes (int, optional): The number of classes for classification. Only relevant if `inclue_top`
            is `True`.
        use_batch_norm (bool, optional): If `True`, adds a batch normalization layer
            after each convolutional layer.
        use_max_pooling (tuple, optional): A tuple of four booleans which define whether max pooling is being
            performed after the stages 1, 2, 3, and 4, respectively.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        verbose (bool, optional):
    '''

    build_function = get_single_block_resnet_build_function(kernel_type=kernel_type,
                                                            kernel_size=kernel_size,
                                                            h=h,
                                                            gamma=gamma,
                                                            num_stages=num_stages,
                                                            blocks_per_stage=blocks_per_stage,
                                                            filters_per_block=filters_per_block,
                                                            strides=strides,
                                                            include_top=include_top,
                                                            fc_activation=fc_activation,
                                                            num_classes=num_classes,
                                                            use_batch_norm=use_batch_norm,
                                                            use_max_pooling=use_max_pooling,
                                                            l2_regularization=l2_regularization,
                                                            subtract_mean=subtract_mean,
                                                            divide_by_stddev=divide_by_stddev,
                                                            verbose=verbose)

    input_tensor = tf.keras.layers.Input(shape=image_shape)

    return build_function(input_tensor)

def get_single_block_resnet_build_function(kernel_type='antisymmetric',
                                           kernel_size=3,
                                           h=1.0,
                                           gamma=0.0,
                                           num_stages=5,
                                           blocks_per_stage=[3, 4, 6, 3],
                                           filters_per_block=[64, 128, 256, 512],
                                           strides=[(2,2),(2,2),(2,2),(2,2)],
                                           include_top=True,
                                           fc_activation='softmax',
                                           num_classes=None,
                                           use_batch_norm=False,
                                           use_max_pooling=[False, False, False, False],
                                           l2_regularization=0.0,
                                           subtract_mean=None,
                                           divide_by_stddev=None,
                                           verbose=False):

    if include_top and (num_classes is None):
        raise ValueError("You must pass a positive integer for `num_classes` if `include_top` is `True`.")

    name = 'single_block_resnet'

    if kernel_type == 'antisymmetric':
        antisymmetric = True
        name += '_antisymmetric'
    else:
        antisymmetric = False
        name += '_regular'

    if not (subtract_mean is None):
        subtract_mean = np.array(subtract_mean)

    if not (divide_by_stddev is None):
        divide_by_stddev = np.array(divide_by_stddev)

    def _build_function(input_tensor):
        '''
        Build the network given an input tensor.
        '''

        input_shape = list(input_tensor.shape)

        # The following identity layer is only needed so that the subsequent lambda layers can be optional.
        x1 = Lambda(lambda x: x, output_shape=input_shape, name='identity_layer')(input_tensor)
        if not (subtract_mean is None):
            x1 = Lambda(lambda x: x - subtract_mean, output_shape=input_shape, name='input_mean_shift')(x1)
        if not (divide_by_stddev is None):
            x1 = Lambda(lambda x: x / divide_by_stddev, output_shape=input_shape, name='input_scaling')(x1)

        # Stage 1
        if verbose: print("Building stage 1")
        x = Conv2D(filters=filters_per_block[0],
                   kernel_size=kernel_size,
                   strides=strides[0],
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_regularization),
                   name='conv1')(x1)
        if use_batch_norm:
            x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = Activation('relu')(x)

        # Stage 2 to num_stages.
        for s in range(num_stages-1):
            # This is stage s+2.
            if use_max_pooling[s]:
                x = MaxPooling2D(pool_size=(2, 2), strides=None, name='stage{}_pooling'.format(s+2))(x)
            if (s == 0) and not use_max_pooling[s]: # In the second stage we only need identity blocks because the number of filters remains constant.
                for b in range(0, blocks_per_stage[s]):
                    if verbose: print("Building identity block {}-{}".format(s+2, b+1))
                    x = single_layer_identity_block(x, kernel_size, antisymmetric, use_batch_norm, stage=s+2, block=b, h=h, gamma=gamma, kernel_regularizer=l2(l2_regularization))
            else:
                if not use_max_pooling[s] and (filters_per_block[s] == filters_per_block[s-1]) and (strides[s] == (1,1)): # Output shape does not change.
                    for b in range(0, blocks_per_stage[s]):
                        if verbose: print("Building identity block {}-{}".format(s+2, b+1))
                        x = single_layer_identity_block(x, kernel_size, antisymmetric, use_batch_norm, stage=s+2, block=b, h=h, gamma=gamma, kernel_regularizer=l2(l2_regularization))
                else:
                    if verbose: print("Building conv block {}-{}".format(s+2, 1))
                    x = single_layer_conv_block(x, kernel_size, filters_per_block[s], strides[s], use_batch_norm, stage=s+2, block=0, kernel_regularizer=l2(l2_regularization))
                    for b in range(1, blocks_per_stage[s]):
                        if verbose: print("Building identity block {}-{}".format(s+2, b))
                        x = single_layer_identity_block(x, kernel_size, antisymmetric, use_batch_norm, stage=s+2, block=b, h=h, gamma=gamma, kernel_regularizer=l2(l2_regularization))

        if include_top:
            x = GlobalAveragePooling2D(name='global_average_pooling')(x)
            x = Dense(num_classes, activation=fc_activation, kernel_initializer='he_normal', kernel_regularizer=l2(l2_regularization), name='fc')(x)

        # Create the model.
        model = tf.keras.models.Model(input_tensor, x, name=name)

        return model

    return _build_function

def build_resnet(image_shape,
                 kernel_type='antisymmetric',
                 include_top=True,
                 fc_activation='softmax',
                 num_classes=None,
                 l2_regularization=0.0,
                 subtract_mean=None,
                 divide_by_stddev=None,
                 version=1,
                 preset=None,
                 blocks_per_stage=[3, 4, 6, 3],
                 filters_per_block=[[64, 64, 256],
                                    [128, 128, 512],
                                    [256, 256, 1024],
                                    [512, 512, 2048]],
                 use_batch_norm=True):
    '''
    Build a ResNet with the regular bottleneck block architecture. The overall ResNet is composed
    of five stages of ResNet blocks. Each stage consists of one or more identical blocks.

    Arguments:
        image_shape (tuple): A tuple `(height, width, channels)` of three integers representing the size
            and number of channels of the image input.
        kernel_type (str, optional): If 'antisymmetric', will build a ResNet in which
            all 3-by-3 convolution kernels are anti-centrosymmetric.
        include_top (bool, optional): If `False`, the output of the last convolutional layer is the model output.
            Otherwise, an average pooling layer and a fully connected layer with `num_classes` outputs followed
            by a softmax activation ayer will be added, the output of the latter of which will be the model output.
        fc_activation (str, optional): The activation function to use for the very last layer of the network,
            i.e. the dense layer. Can be any valid Keras activation function name, e.g. 'softmax' for classification.
            If this is `None`, no activation will be applied to the dense layer. Only relevant if `include_top`
            is True.
        num_classes (int, optional): The number of classes for classification. Only relevant if `inclue_top`
            is `True`.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        version (float, optional): A value from the set {1, 1.5}, the version of the ResNet.
            The different versions are defined as follows:

            v1: The striding of the conv block is performed by the first 1-by-1 convolution.
                The order of operations of each convolutional layer is (conv, BN, ReLU).
            v1.5: The striding of the conv block is performed by the 3-by-3 convolution.
                The order of operations of each convolutional layer is (conv, BN, ReLU).
            v2 (currently not supported): The striding of the conv block is performed by
                the first 3-by-3 convolution. The order of operations of each convolutional layer
                is (BN, ReLU, conv).

            The reasoning for introducing v1.5 is that is has been reported that performing the
            striding (i.e. the spatial dimensionality reduction) in the 3-by-3 convolution instead
            of the 1-by-1 convolution results in higher and more stable accuracy in fewer epochs
            than the original v1 and has shown to scale to higher batch sizes with minimal degradation
            in accuracy. However, it has also been reported that v1.5 requires ~12% more compute to
            train and has 6% reduced throughput for inference compared to v1.
        preset (str, optional): Must be either one of 'resnet50', 'resnet101', and 'resnet152', in which case
            the respective ResNet architecture will be built, or `None` in which case no preset will be used.
            If a preset is passed, then all of the subsequent arguments will be overwritten by the preset, i.e.
            their passed values become irrelevant.
        blocks_per_stage (tuple, optional): A tuple of four positive integers representing the number of
            ResNet blocks for the stages 2, 3, 4, and 5 of the ResNet. The default configuration produces
            a ResNet50. To produce a ResNet101 or ResNet152, use the configurations `[3, 4, 23, 3]` and
            `[3, 8, 36, 3]`, respectively.
        filters_per_block (tuple, optional): A tuple of four 3-tuples. The four tuples represent the number of
            filters to be used for the convolutional layers of the blocks in each of the stages 2, 3, 4, and 5
            of the ResNet. Each 3-tuple defines the number of filters to be used for the three convolutional
            layers (1-by-1, 3-by-3, 1-by-1) of the blocks in the respective stage.
        use_batch_norm (bool, optional): If `True`, adds a batch normalization layer directly
            after each convolutional layer.
    '''

    build_function = get_resnet_build_function(kernel_type=kernel_type,
                                               include_top=include_top,
                                               fc_activation=fc_activation,
                                               num_classes=num_classes,
                                               l2_regularization=l2_regularization,
                                               subtract_mean=subtract_mean,
                                               divide_by_stddev=divide_by_stddev,
                                               version=version,
                                               preset=preset,
                                               blocks_per_stage=blocks_per_stage,
                                               filters_per_block=filters_per_block,
                                               use_batch_norm=use_batch_norm)

    input_tensor = tf.keras.layers.Input(shape=image_shape)

    return build_function(input_tensor)

def get_resnet_build_function(kernel_type='antisymmetric',
                              include_top=True,
                              fc_activation='softmax',
                              num_classes=None,
                              l2_regularization=0.0,
                              subtract_mean=None,
                              divide_by_stddev=None,
                              version=1,
                              preset=None,
                              blocks_per_stage=[3, 4, 6, 3],
                              filters_per_block=[[64, 64, 256],
                                                [128, 128, 512],
                                                [256, 256, 1024],
                                                [512, 512, 2048]],
                              use_batch_norm=True):

    if include_top and (num_classes is None):
        raise ValueError("You must pass a positive integer for `num_classes` if `include_top` is `True`.")

    name = 'resnet'

    if not (preset is None):
        if preset == 'resnet50':
            blocks_per_stage = [3, 4, 6, 3]
            filters_per_block = [[64, 64, 256],
                                 [128, 128, 512],
                                 [256, 256, 1024],
                                 [512, 512, 2048]]
            use_batch_norm = True
            name += '50'
        elif preset == 'resnet101':
            blocks_per_stage = [3, 4, 23, 3]
            filters_per_block = [[64, 64, 256],
                                 [128, 128, 512],
                                 [256, 256, 1024],
                                 [512, 512, 2048]]
            use_batch_norm = True
            name += '101'
        elif preset == 'resnet152':
            blocks_per_stage = [3, 8, 36, 3]
            filters_per_block = [[64, 64, 256],
                                 [128, 128, 512],
                                 [256, 256, 1024],
                                 [512, 512, 2048]]
            use_batch_norm = True
            name += '152'
        else:
            raise ValueError("`preset` must be either `None` or one of 'resnet50', 'resnet101', and 'resnet152', but you passed `preset={}`.".format(preset))

    if kernel_type == 'antisymmetric':
        antisymmetric = True
        name += '_antisymmetric'
    else:
        antisymmetric = False
        name += '_regular'

    if not (subtract_mean is None):
        subtract_mean = np.array(subtract_mean)

    if not (divide_by_stddev is None):
        divide_by_stddev = np.array(divide_by_stddev)

    def _build_function(input_tensor):
        '''
        Build the network given an input tensor.
        '''

        input_shape = list(input_tensor.shape)

        # The following identity layer is only needed so that the subsequent lambda layers can be optional.
        x1 = Lambda(lambda x: x, output_shape=input_shape, name='identity_layer')(input_tensor)
        if not (subtract_mean is None):
            x1 = Lambda(lambda x: x - subtract_mean, output_shape=input_shape, name='input_mean_shift')(x1)
        if not (divide_by_stddev is None):
            x1 = Lambda(lambda x: x / divide_by_stddev, output_shape=input_shape, name='input_scaling')(x1)

        # Stage 1
        x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x1)
        x = Conv2D(filters=64,
                   kernel_size=(7, 7),
                   strides=(2, 2),
                   padding='valid',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_regularization),
                   name='conv1')(x)
        if use_batch_norm:
            x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='stage1_pooling')(x)

        # Stage 2
        x = bottleneck_conv_block(x, 3, filters_per_block[0], antisymmetric, use_batch_norm, stage=2, block=0, version=version, strides=(1,1), kernel_regularizer=l2(l2_regularization))
        for i in range(1, blocks_per_stage[0]):
            x = bottleneck_identity_block(x, 3, filters_per_block[0], antisymmetric, use_batch_norm, stage=2, block=i, kernel_regularizer=l2(l2_regularization))

        # Stage 3
        x = bottleneck_conv_block(x, 3, filters_per_block[1], antisymmetric, use_batch_norm, stage=3, block=0, version=version, strides=(2,2), kernel_regularizer=l2(l2_regularization))
        for i in range(1, blocks_per_stage[1]):
            x = bottleneck_identity_block(x, 3, filters_per_block[1], antisymmetric, use_batch_norm, stage=3, block=i, kernel_regularizer=l2(l2_regularization))

        # Stage 4
        x = bottleneck_conv_block(x, 3, filters_per_block[2], antisymmetric, use_batch_norm, stage=4, block=0, version=version, strides=(2,2), kernel_regularizer=l2(l2_regularization))
        for i in range(1, blocks_per_stage[2]):
            x = bottleneck_identity_block(x, 3, filters_per_block[2], antisymmetric, use_batch_norm, stage=4, block=i, kernel_regularizer=l2(l2_regularization))

        # Stage 5
        x = bottleneck_conv_block(x, 3, filters_per_block[3], antisymmetric, use_batch_norm, stage=5, block=0, version=version, strides=(2,2), kernel_regularizer=l2(l2_regularization))
        for i in range(1, blocks_per_stage[3]):
            x = bottleneck_identity_block(x, 3, filters_per_block[3], antisymmetric, use_batch_norm, stage=5, block=i, kernel_regularizer=l2(l2_regularization))

        if include_top:
            x = GlobalAveragePooling2D(name='global_average_pooling')(x)
            x = Dense(num_classes, activation=fc_activation, kernel_initializer='he_normal', kernel_regularizer=l2(l2_regularization), name='fc')(x)

        # Create the model.
        model = tf.keras.models.Model(input_tensor, x, name=name)

        return model

    return _build_function
