'''
A class to a tf.data.Dataset from TFRecords.

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
import os
import glob

class TFRecordDatasetCreator:

    def __init__(self,
                 tfrecord_paths,
                 feature_schema,
                 batch_size,
                 preprocessors=None,
                 num_epochs=None,
                 shuffle=True,
                 shuffle_buffer_size=None,
                 num_parallel_reads=None,
                 num_parallel_calls=None):

        if len(tfrecord_paths) != len(set(tfrecord_paths)):
            raise ValueError('tfrecord_paths {} are not unique.'.format(tfrecord_paths))
        if len(tfrecord_paths) == 0:
            raise ValueError('No tfrecords_paths specified.')
        if shuffle and shuffle_buffer_size is None:
            raise ValueError('If using shuffle, please specify a shuffle buffer size.')

        self.tfrecord_paths = [os.path.abspath(tfrecord_path) for tfrecord_path in tfrecord_paths]
        self.feature_schema = feature_schema
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_parallel_reads = num_parallel_reads
        self.num_parallel_calls = num_parallel_calls

    def _process(self):

        # Shuffle the TFRecord file names.
        self.dataset = tf.data.Dataset.from_tensor_slices(self.tfrecord_paths)
        if self.shuffle:
            self.dataset = self.dataset.shuffle(len(self.tfrecord_paths))

        # Create one unified dataset from the TFRecord files.
        self.dataset = self.dataset.flat_map(lambda filename: tf.data.TFRecordDataset(filename,
                                                                                      num_parallel_reads=self.num_parallel_reads))

        # Deserialize tf.Example objects.
        self.dataset = self.dataset.map(lambda single_example_proto: tf.parse_single_example(single_example_proto, self.feature_schema),
                                        num_parallel_calls=self.num_parallel_calls)

        # Apply any preprocessing.
        for preprocessor in self.preprocessors or []:
            self.dataset = preprocessor(self.dataset)

        # Shuffle the dataset.
        if self.shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=self.shuffle_buffer_size)

    def create_dataset(self):

        self._process()
        dataset = self.dataset.batch(self.batch_size)
        return dataset

    def create_input_function(self):

        def _input_function():

            self._process()
            dataset = self.dataset.repeat(self.num_epochs)
            dataset = dataset.batch(self.batch_size)

            iterator = dataset.make_one_shot_iterator()
            return iterator

        return _input_function

    def create_generator(self):

        self._process()
        dataset = self.dataset.batch(self.batch_size)

        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            try:
                while True:
                    yield sess.run(batch)

            except tf.errors.OutOfRangeError:
                print("The generator has iterated over the dataset once and is no longer usable.")

def get_tfrecord_paths(directory, extension='tfrecord'):

    return glob.glob(os.path.join(directory, '*.'+extension))
