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

class Conv2DAntisymmetric(tf.keras.layers.Layer):
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
                 kernel_size,
                 strides=(1, 1),
                 use_bias=True,
                 kernel_initializer='he_normal',
                 kernel_regularizer=None,
                 antisymmetric=True,
                 **kwargs):
        '''
        Arguments:
            kernel_size (int): The size of the two spatial dimensions of the quadratic kernel.
            num_filters (int, optional): The number of convolution filters to use. If `None`,
                the number of output filters will be the same as the number of input filters.
                Note that this property is necessary in order to warrant an anti-symmetric
                convolution matrix.
            strides (tuple, optional): The strides along the height and width dimensions
                for the convolution.
            use_bias (bool, optional): Whether or not to add a bias term.
        '''

        super(Conv2DAntisymmetric, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.antisymmetric = antisymmetric

    def build(self, input_shape):

        # The number of output channels must be identical to the number of input channels,
        # otherwise the concept of (anti-) symmetry for the convolution matrix is undefined.
        self.num_channels = int(input_shape[-1]) # Assuming channels-last format.

        ########################################################################
        # Maybe create the kernel initializer.
        ########################################################################

        if self.kernel_initializer == 'he_normal':
            self.kernel_initializer = tf.initializers.truncated_normal(mean=0.0,
                                                                       stddev=tf.sqrt(2/(self.kernel_size * self.kernel_size * self.num_channels)),
                                                                       dtype=self.dtype)

        ########################################################################
        # Build the (anti-) centrosymmetric convolution kernel.
        ########################################################################

        exchange_matrix = tf.constant(np.eye(self.kernel_size)[::-1], dtype=self.dtype)

        self.single_output_kernels = []
        self.independent_kernels = []

        tr = trange(self.num_channels, file=sys.stdout)
        tr.set_description('Building output channel filters for layer')

        for o in tr:
            # Build the centrosymmetric kernel for this output channel.
            centrosymmetric_kernel = self._get_centrosymmetric_matrix()
            # Independent kernels for this output channel
            num_independent = self.num_channels - o - 1 # The number of independent input channels for this output channel.
            if num_independent > 0:
                independent_kernel = self.add_weight(name='input_kernels_for_output_kernel_{}'.format(o),
                                                     shape=[self.kernel_size, self.kernel_size, num_independent, 1],
                                                     dtype=self.dtype,
                                                     initializer=self.kernel_initializer,
                                                     regularizer=self.kernel_regularizer,
                                                     trainable=self.trainable)
                # Concatenate.
                single_output_kernel = tf.concat([centrosymmetric_kernel, independent_kernel], axis=2)
            else: # This case is for the very last output channel.
                single_output_kernel = centrosymmetric_kernel
            # Build the dependent kernels for this output channel.
            for i in range(o):
                # Note: Multiplying by the exchange matrix to create the dependent kernels is not efficient,
                # but it makes creating the (anti-) symmetric kernels simpler and faster to initialize.
                # In a production system, however, the kernels should be transformed directly,
                # introducing unnecessary arithmetic operations.
                centro_transpose = tf.expand_dims(tf.expand_dims(-tf.matmul(exchange_matrix, tf.matmul(self.independent_kernels[-(i+1)][:,:,i,0], exchange_matrix)), axis=-1), axis=-1)
                single_output_kernel = tf.concat([centro_transpose, single_output_kernel], axis=2)
            self.single_output_kernels.append(single_output_kernel)
            if num_independent > 0:
                self.independent_kernels.append(independent_kernel)

        self.kernel = tf.concat(self.single_output_kernels, axis=3, name='kernel')

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

        super(Conv2DAntisymmetric, self).build(input_shape)

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
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'antisymmetric': self.antisymmetric
        }
        base_config = super(Conv2DAntisymmetric, self).get_config()
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

    def _get_centrosymmetric_matrix(self, prefix=None):
        '''
        Returns a tensor containing an (anti-) centrosymmetric matrix.

        Arguments:
            prefix (str, optional): A prefix string to use for the names of the individual
                scalar variables that make up the matrix.
        '''

        if prefix is None: prefix = 'centro_sym'

        # We'll create a (anti-) centrosymmetric matrix from individual scalar TensorFlow variables.
        # We'll first store the references to these scalar variables in a Numpy array, then concatenate
        # them all to one tensor.
        variable_array = np.full(fill_value=None, shape=(self.kernel_size, self.kernel_size)) # Store references to scalar variables here.
        for i in range(self.kernel_size):
            for j in range(i, self.kernel_size): # Only half of the matrix entries are free variables.
                if j > i or (j == i and i <= self.kernel_size // 2 - 1):
                    variable = self.add_weight(name='{}_{}_{}'.format(prefix, i, j),
                                               shape=[1, 1, 1, 1],
                                               dtype=self.dtype,
                                               initializer=self.kernel_initializer,
                                               regularizer=self.kernel_regularizer,
                                               trainable=self.trainable)
                    # This is the definition of (anti-) centrosymmetry:
                    variable_array[i, j] = variable
                    variable_array[self.kernel_size - 1 - i, self.kernel_size - 1 - j] = -variable if self.antisymmetric else variable
                elif j == i and i == self.kernel_size // 2 and self.kernel_size % 2 == 1: # For matrices of odd size, this is the central element.
                    if self.antisymmetric:
                        # For the anti-centrosymmetric case, the center element must be zero.
                        # This also implies that this element must be non-trainable.
                        variable_array[i, j] = self.add_weight(name='{}_{}_{}'.format(prefix, i, j),
                                                               shape=[1, 1, 1, 1],
                                                               dtype=self.dtype,
                                                               initializer=tf.initializers.zeros(dtype=self.dtype),
                                                               regularizer=None,
                                                               trainable=False) # Must be non-trainable.
                    else:
                        variable_array[i, j] = self.add_weight(name='{}_{}_{}'.format(prefix, i, j),
                                                               shape=[1, 1, 1, 1],
                                                               dtype=self.dtype,
                                                               initializer=self.kernel_initializer,
                                                               regularizer=self.kernel_regularizer,
                                                               trainable=self.trainable)

        variable_array = variable_array.tolist()
        variable_array = [tf.concat(var_list, axis=1) for var_list in variable_array]
        tensor = tf.concat(variable_array, axis=0)

        return tensor
