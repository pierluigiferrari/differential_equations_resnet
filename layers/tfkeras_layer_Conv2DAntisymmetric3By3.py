'''
A custom tf.keras 2D convolutional layer with anti-centrosymmetric 3-by-3 kernels.

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
from tqdm import trange
import sys

class Conv2DAntisymmetric3By3(tf.keras.layers.Layer):
    '''
    A custom tf.keras 2D convolutional layer with anti-centrosymmetric 3-by-3 kernels.

    This layer works just like a regular 2D convolutional layer, except that its
    kernels take a particular form, namely they are anti-centrosymmetric with respect
    to their spatial dimensions (height and width). That is, the kernel takes the
    following form along its spatial dimensions:

    [[ a,  b,  c],
     [ d,  0, -d],
     [-c, -b, -a]],

    where a, b, c, d are real numbers. The kernel is He-initialized.

    As is well-known, discrete convolution of a kernel with an input can be formulated
    as a matrix-vector multiplication where the convolution kernel is transformed
    into a doubly-blocked Toeplitz convolution matrix. In the case of a zero-padded
    convolution with stride (1,1) and dilation (1,1), this convolution matrix is
    quadratic. Anti-centrosymmetry of the convolution kernel is then equivalent to
    anti-symmetry of the respective convolution matrix. Anti-symmetry of the
    convolution matrix provides certain stability properties
    (see https://arxiv.org/abs/1705.03341).

    This convolution layer underlies the following limitations:

        1. Strided convolution is not recommended (necessary to ensure
           that the convolution matrix is anti-symmetric)
        2. Dilated convolution is not supported (necessary to ensure
           that the convolution matrix is anti-symmetric)
        3. Currently, only 3-by-3 kernels are supported (in general, any kernel
           size could be supported, but it involves some effort to support general
           kernel sizes)
    '''

    def __init__(self,
                 gamma=0.0,
                 strides=(1, 1),
                 use_bias=True,
                 kernel_initializer='he_normal',
                 kernel_regularizer=None,
                 **kwargs):
        '''
        Arguments:
            num_filters (int, optional): The number of convolution filters to use. If `None`,
                the number of output filters will be the same as the number of input filters.
                Note that this property is necessary in order to warrant an anti-symmetric
                convolution matrix.
            strides (tuple, optional): The strides along the height and width dimensions
                for the convolution.
            use_bias (bool, optional): Whether or not to add a bias term.
        '''

        super(Conv2DAntisymmetric3By3, self).__init__(**kwargs)
        self.gamma = gamma
        self.strides = strides
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):

        # The number of output channels must be identical to the number of input channels,
        # otherwise the concept of (anti-) symmetry for the convolution matrix is undefined.
        self.num_channels = int(input_shape[-1]) # Assuming channels-last format.

        ########################################################################
        # Maybe create the kernel initializer.
        ########################################################################

        if self.kernel_initializer == 'he_normal':
            self.kernel_initializer = tf.initializers.truncated_normal(mean=0.0,
                                                                       stddev=tf.sqrt(2/(3 * 3 * self.num_channels)),
                                                                       dtype=self.dtype)

        ########################################################################
        # Build the (anti-) centrosymmetric convolution kernel.
        ########################################################################

        self.single_output_kernels = []
        self.independent_kernels = []
        self.anti_centrosymmetric_transpose = []

        tr = trange(self.num_channels, file=sys.stdout)
        tr.set_description('Building output channel filters for layer')

        # Build the centrosymmetric kernel for all `n` channels, i.e. the (i,i) blocks of the convolution matrix
        # for i = 1,...,n.
        self.anti_centrosymmetric_kernel = self._get_anti_centrosymmetric_kernel() # Shape: (3, 3, 1, num_channels)

        for o in tr:
            # Independent kernels for this output channel
            num_independent = self.num_channels - o - 1 # The number of independent input channels for this output channel.
            if num_independent > 0:
                independent_kernel = self.add_weight(name='input_kernels_for_output_kernel_{}'.format(o),
                                                     shape=[3, 3, num_independent],
                                                     dtype=self.dtype,
                                                     initializer=self.kernel_initializer,
                                                     regularizer=self.kernel_regularizer,
                                                     trainable=self.trainable)
                anti_centrosymmetric_transpose = self._get_anti_centrosymmetric_transpose(independent_kernel)
                # Get a slice from the anti-centrosymmetric kernel.
                anti_centrosymmetric_slice = self.anti_centrosymmetric_kernel[:, :, :, o]
                # Concatenate.
                single_output_kernel = tf.concat([anti_centrosymmetric_slice, independent_kernel], axis=-1)
            else: # This case is for the very last output channel.
                single_output_kernel = self.anti_centrosymmetric_kernel[:, :, :, o]
            # Concatenate the dependent and independent kernels for this output channel.
            for i in range(o):
                centro_transpose = tf.expand_dims(self.anti_centrosymmetric_transpose[-(i+1)][:, :, i], axis=-1)
                single_output_kernel = tf.concat([centro_transpose, single_output_kernel], axis=-1)
            self.single_output_kernels.append(single_output_kernel)
            if num_independent > 0:
                self.independent_kernels.append(independent_kernel)
                self.anti_centrosymmetric_transpose.append(anti_centrosymmetric_transpose)

        self.kernel = tf.stack(self.single_output_kernels, axis=-1, name='kernel')

        ########################################################################
        # Maybe create a bias vector.
        ########################################################################

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=[self.num_channels,],
                                        dtype=self.dtype,
                                        initializer=tf.initializers.zeros(dtype=self.dtype),
                                        regularizer=None,
                                        trainable=self.trainable)

        super(Conv2DAntisymmetric3By3, self).build(input_shape)

    def call(self, input_tensor):

        output_tensor = tf.nn.conv2d(input_tensor,
                                     self.kernel,
                                     strides=[1, self.strides[0], self.strides[1], 1],
                                     padding="SAME",
                                     use_cudnn_on_gpu=True,
                                     data_format='NHWC',
                                     dilations=[1, 1, 1, 1],
                                     name=None)

        if self.use_bias:
            output_tensor += self.bias

        return output_tensor

    def compute_output_shape(self, input_shape):

        return input_shape

    def get_config(self):

        config = {
            'strides': self.strides,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer
        }
        base_config = super(Conv2DAntisymmetric3By3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_kernel(self):
        '''
        Return this layer's convolution kernel.

        The Layer object's standard `get_weights()` method is inconvenient for this
        layer because its kernel consists of multiple individual weights variables,
        which the `get_weights()` method will return individually instead of as
        one composed kernel. In order to retrieve the kernel, use this method
        instead.
        '''

        return tf.keras.backend.batch_get_value([self.kernel])[0]

    def get_bias(self):
        '''
        Return this layer's bias vector.

        This method exists for the same reason as the `get_kernel()` method.
        '''

        return tf.keras.backend.batch_get_value([self.bias])[0]

    def _get_anti_centrosymmetric_kernel(self):
        '''
        Returns a tensor containing an (anti-) centrosymmetric matrix.

        Arguments:
            prefix (str, optional): A prefix string to use for the names of the individual
                scalar variables that make up the matrix.
        '''

        self.a = self.add_weight(name='a',
                                 shape=[1, 1, 1, self.num_channels],
                                 dtype=self.dtype,
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 trainable=self.trainable)

        self.b = self.add_weight(name='b',
                                 shape=[1, 1, 1, self.num_channels],
                                 dtype=self.dtype,
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 trainable=self.trainable)

        self.c = self.add_weight(name='c',
                                 shape=[1, 1, 1, self.num_channels],
                                 dtype=self.dtype,
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 trainable=self.trainable)

        self.d = self.add_weight(name='d',
                                 shape=[1, 1, 1, self.num_channels],
                                 dtype=self.dtype,
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 trainable=self.trainable)

        # 'e' constitutes the center element of the kernel, which must be constant (usually zero) and non-trainable.
        self.e = tf.fill(dims=[1, 1, 1, self.num_channels],
                         value=self.gamma,
                         name='e')

        '''
        self.e = self.add_weight(name='e',
                                 shape=[1, 1, 1, self.num_channels],
                                 dtype=self.dtype,
                                 initializer=tf.initializers.constant(value=self.gamma, dtype=self.dtype),
                                 regularizer=None,
                                 trainable=False)
        '''

        # The remaining elements of the kernel are just the additive inverses of the previous elements.
        self.f = -self.d
        self.g = -self.c
        self.h = -self.b
        self.i = -self.a

        # Put the nine individual spatial kernel elements together into one 3x3 kernel.
        row1 = tf.concat(values=[self.a,self.b,self.c], axis=1)
        row2 = tf.concat(values=[self.d,self.e,self.f], axis=1)
        row3 = tf.concat(values=[self.g,self.h,self.i], axis=1)
        return tf.concat(values=[row1,
                                 row2,
                                 row3],
                         axis=0,
                         name='anti_centrosym_kernel')

    def _get_anti_centrosymmetric_transpose(self, tensor):

        a = -tensor[0, 0, :]
        b = -tensor[0, 1, :]
        c = -tensor[0, 2, :]
        d = -tensor[1, 0, :]
        e = -tensor[1, 1, :]
        f = -tensor[1, 2, :]
        g = -tensor[2, 0, :]
        h = -tensor[2, 1, :]
        i = -tensor[2, 2, :]

        # Put the nine individual spatial kernel elements together into one 3x3 kernel.
        row1 = tf.stack(values=[i, h, g], axis=0)
        row2 = tf.stack(values=[f, e, d], axis=0)
        row3 = tf.stack(values=[c, b, a], axis=0)
        return tf.stack(values=[row1, row2, row3], axis=0)
