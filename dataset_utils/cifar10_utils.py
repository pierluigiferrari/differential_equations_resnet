'''
Utilities for the Python version of the CIFAR-10 dataset.

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
import numpy as np
import pickle
import os

def build_cifar10_dataset(cifar10_directory):
    '''
    Load and transform the Python version of the CIFAR-10 dataset given the directory
    path to the extracted data set downloaded from https://www.cs.toronto.edu/~kriz/cifar.html.

    Arguments:
        cifar10_directory (str): The path to the extracted data set downloaded from
            https://www.cs.toronto.edu/~kriz/cifar.html.

    Returns:
        Five objects, in this order:
        - A Numpy array containing all 50,000 training images
        - A Numpy array containing all 50,000 labels for the training images
        - A Numpy array containing all 10,000 test images
        - A Numpy array containing all 10,000 labels for the test images
        - A list containing the names of the ten classes
    '''

    train_pickle_filenames = ['data_batch_1',
                              'data_batch_2',
                              'data_batch_3',
                              'data_batch_4',
                              'data_batch_5']
    test_pickle_filename = 'test_batch'
    label_names_pickle_filename = 'batches.meta'

    # Compile the training data set files.
    train_images = []
    train_labels = []
    for filename in train_pickle_filenames:
        dictionary = unpickle(os.path.join(cifar10_directory, filename))
        train_images.append(dictionary[b'data'])
        train_labels.append(dictionary[b'labels'])
    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # Compile the test data set files.
    dictionary = unpickle(os.path.join(cifar10_directory, test_pickle_filename))
    test_images = dictionary[b'data']
    test_labels = np.asarray(dictionary[b'labels'])

    # Transform the images from (*,3072) to (*, 32, 32, 3).
    train_images = np.reshape(train_images, [50000, 3, 32, 32])
    train_images = np.transpose(train_images, [0, 2, 3, 1])
    test_images = np.reshape(test_images, [10000, 3, 32, 32])
    test_images = np.transpose(test_images, [0, 2, 3, 1])

    # Get the label names.
    dictionary = unpickle(os.path.join(cifar10_directory, label_names_pickle_filename))
    label_names = [str(bytestr, 'utf-8') for bytestr in dictionary[b'label_names']]

    return train_images, train_labels, test_images, test_labels, label_names

def unpickle(filename):
    with open(filename, 'rb') as f:
        dictionary = pickle.load(f, encoding='bytes')
    return dictionary
