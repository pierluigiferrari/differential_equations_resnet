'''
Functions to create a tf.data.Dataset from NumPy arrays.

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

def create_tf_dataset_from_arrays(features,
                                  labels,
                                  batch_size,
                                  preprocessors=None,
                                  repeat=True,
                                  num_epochs=None,
                                  shuffle=True,
                                  prefetch=None):

    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels.shape[0]

    features_placeholder = tf.placeholder(dtype=features.dtype, shape=features.shape, name='features_placeholder')
    labels_placeholder = tf.placeholder(dtype=labels.dtype, shape=labels.shape, name='labels_placeholder')

    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))

    # Apply any preprocessing.
    for preprocessor in preprocessors or []:
        dataset = preprocessor(dataset)

    # Shuffle the dataset.
    if shuffle:
        dataset = dataset.shuffle(buffer_size=features.shape[0],
                                  reshuffle_each_iteration=True)

    if repeat:
        # Repeat the dataset.
        dataset = dataset.repeat(num_epochs)

    if not (batch_size is None):
        dataset = dataset.batch(batch_size)

    if not prefetch is None:
        dataset = dataset.prefetch(buffer_size=prefetch)

    return dataset, features_placeholder, labels_placeholder
