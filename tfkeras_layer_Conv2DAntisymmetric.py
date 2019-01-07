'''
A custom tf.keras 2D convolutional layer with skew-centrosymmetric 3-by-3 kernels.

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

class Conv2DAntisymmetric(tf.keras.layers.Layer):
    '''
    A custom tf.keras 2D convolutional layer with skew-centrosymmetric 3-by-3 kernels.

    This layer works just like a regular 2D convolutional layer, except that its
    kernels take a particular form, namely they are skew-centrosymmetric.

    As is well-known, discrete convolution of a kernel with an input can be formulated
    as a matrix-vector multiplication where the convolution kernel is transformed
    into a doubly-blocked Toeplitz convolution matrix. In the case of a zero-padded
    convolution with stride (1,1) and dilation (1,1), this convolution matrix is
    quadratic. Anti-centrosymmetry of the convolution kernel is then equivalent to
    anti-symmetry of the respective convolution matrix. Anti-symmetry of the
    convolution matrix provides certain stability properties
    (see https://arxiv.org/abs/1705.03341).

    This convolution layer underlies certain limitations:

        1. Strided convolution is not supported (necessary to ensuring
           that the convolution matrix is anti-symmetric)
        2. Dilated convolution is not supported (necessary to ensuring
           that the convolution matrix is anti-symmetric)
        3. Currently, only 3-by-3 kernels are supported (in general, any kernel
           size could be supported, but it involves some effort to support general
           kernel sizes)
    '''

    def __init__(self,
                 num_filters,
                 use_bias=True,
                 **kwargs):

        super(Conv2DAntisymmetric, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.use_bias = use_bias

    def build(self, input_shape):

        kernel_height = 3
        kernel_width  = 3
        dtype=tf.float32
        in_channels = int(input_shape[-1])

        self.a = self.add_weight(name='a',
                                 shape=[1,1,in_channels,self.num_filters],
                                 dtype=dtype,
                                 initializer=tf.initializers.truncated_normal(mean=0.0,
                                                                              stddev=tf.sqrt(2/(kernel_height * kernel_width * in_channels)),
                                                                              dtype=dtype),
                                 regularizer=None,
                                 trainable=self.trainable)

        self.b = self.add_weight(name='b',
                                 shape=[1,1,in_channels,self.num_filters],
                                 dtype=dtype,
                                 initializer=tf.initializers.truncated_normal(mean=0.0,
                                                                              stddev=tf.sqrt(2/(kernel_height * kernel_width * in_channels)),
                                                                              dtype=dtype),
                                 regularizer=None,
                                 trainable=self.trainable)

        self.c = self.add_weight(name='c',
                                 shape=[1,1,in_channels,self.num_filters],
                                 dtype=dtype,
                                 initializer=tf.initializers.truncated_normal(mean=0.0,
                                                                              stddev=tf.sqrt(2/(kernel_height * kernel_width * in_channels)),
                                                                              dtype=dtype),
                                 regularizer=None,
                                 trainable=self.trainable)

        self.d = self.add_weight(name='d',
                                 shape=[1,1,in_channels,self.num_filters],
                                 dtype=dtype,
                                 initializer=tf.initializers.truncated_normal(mean=0.0,
                                                                              stddev=tf.sqrt(2/(kernel_height * kernel_width * in_channels)),
                                                                              dtype=dtype),
                                 regularizer=None,
                                 trainable=self.trainable)

        self.e = self.add_weight(name='e',
                                 shape=[1,1,in_channels,self.num_filters],
                                 dtype=dtype,
                                 initializer=tf.initializers.zeros(dtype=dtype),
                                 regularizer=None,
                                 trainable=False)

        self.f = -self.d
        self.g = -self.c
        self.h = -self.b
        self.i = -self.a

        # Put the nine individual spatial kernel elements together into one 3x3 kernel.
        kernel_row1 = tf.concat(values=[self.a,self.b,self.c],
                                axis=1,
                                name='kernel_row1')

        kernel_row2 = tf.concat(values=[self.d,self.e,self.f],
                                axis=1,
                                name='kernel_row2')

        kernel_row3 = tf.concat(values=[self.g,self.h,self.i],
                                axis=1,
                                name='kernel_row3')

        self.kernel = tf.concat(values=[kernel_row1,
                                        kernel_row2,
                                        kernel_row3],
                                axis=0,
                                name='skew_centrosymmetric_kernel')

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=[self.num_filters,],
                                        dtype=dtype,
                                        initializer=tf.initializers.zeros(dtype=dtype),
                                        regularizer=None,
                                        trainable=self.trainable)

        super(Conv2DAntisymmetric, self).build(input_shape)

    def call(self, input_tensor):

        output_tensor = tf.nn.conv2d(input_tensor,
                                     self.kernel,
                                     strides=[1, 1, 1, 1],
                                     padding="SAME",
                                     use_cudnn_on_gpu=True,
                                     data_format='NHWC',
                                     dilations=[1, 1, 1, 1],
                                     name=None)

        if self.use_bias:
            output_tensor += self.bias

        return output_tensor

    def compute_output_shape(self, input_shape):

        return (input_shape[0], input_shape[1], input_shape[2], self.num_filters)

    def get_config(self):
        config = {
            'num_filters': self.num_filters,
            'use_bias': self.use_bias,
        }
        base_config = super(Conv2DAntisymmetric, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_kernel(self):

        return tf.keras.backend.batch_get_value([self.kernel])[0]

    def get_bias(self):

        return tf.keras.backend.batch_get_value([self.bias])[0]
