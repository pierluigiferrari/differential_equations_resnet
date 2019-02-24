'''
Utility functions to produce antisymmetric 2D-convolution matrices.

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

def get_centrosymmetric_matrix(size,
                               in_channels,
                               rank=4,
                               anti=True,
                               regularizer=None,
                               trainable=True,
                               prefix=None):

    prefix = '' if (prefix is None) else '_' + prefix

    # We'll create a (anti-) centrosymmetric matrix from individual scalar TensorFlow variables.
    # We'll first store the references to these scalar variables in a Numpy array, then concatenate
    # them all to one tensor.
    variable_array = np.full(fill_value=None, shape=(size, size)) # Store references to scalar variables here.
    for i in range(size):
        for j in range(i, size): # Only half of the matrix entries are free variables.
            if j > i or (j == i and i <= size // 2 - 1):
                variable = tf.get_variable(name='centro{}_{}_{}'.format(prefix, i, j),
                                           shape=[1,1] + [1] * (rank - 2),
                                           dtype=tf.float32,
                                           initializer=tf.initializers.truncated_normal(mean=0.0,
                                                                                        stddev=tf.sqrt(2/(size * size * in_channels)),
                                                                                        dtype=tf.float32),
                                           regularizer=regularizer,
                                           trainable=trainable)
                # This is the definition of (anti-) centrosymmetry:
                variable_array[i, j] = variable
                variable_array[size - 1 - i, size - 1 - j] = -variable if anti else variable
            elif j == i and i == size // 2 and size % 2 == 1: # For matrices of odd size, this is the central element.
                if anti:
                    # For the anti-centrosymmetric case, the center element must be zero.
                    # This also implies that this element must be non-trainable.
                    variable_array[i, j] = tf.get_variable(name='centro{}_{}_{}'.format(prefix, i, j),
                                                           shape=[1,1] + [1] * (rank - 2),
                                                           dtype=tf.float32,
                                                           initializer=tf.initializers.zeros(dtype=tf.float32),
                                                           regularizer=regularizer,
                                                           trainable=False) # Must be non-trainable.
                else:
                    variable_array[i, j] = tf.get_variable(name='centro{}_{}_{}'.format(prefix, i, j),
                                                           shape=[1,1] + [1] * (rank - 2),
                                                           dtype=tf.float32,
                                                           initializer=tf.initializers.truncated_normal(mean=0.0,
                                                                                                        stddev=tf.sqrt(2/(size * size * in_channels)),
                                                                                                        dtype=tf.float32),
                                                           regularizer=regularizer,
                                                           trainable=trainable)

    variable_array = variable_array.tolist()
    variable_array = [tf.concat(var_list, axis=1) for var_list in variable_array]
    tensor = tf.concat(variable_array, axis=0)

    return tensor
