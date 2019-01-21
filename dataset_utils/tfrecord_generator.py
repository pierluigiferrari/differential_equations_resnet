'''
A class to generate TFRecords from images.

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
import random

class TFRecordGenerator:

    def __init__(self):
        pass

    def convert(self, directory, tfrecord_file_name, suffix='tfrecord', num_files_per_record=1000, shuffle=True):

        image_paths = get_image_paths(get_subdirectories(directory))

        if shuffle:
            random.shuffle(image_paths)

        num_files_total = len(image_paths)
        start = 0
        end = num_files_per_record
        tfrecord_file_number = 0

        while start < num_files_total:
            with tf.python_io.TFRecordWriter(tfrecord_file_name + '_{:04d}.{}'.format(tfrecord_file_number, suffix)) as writer:
                for image_path in image_paths[start:end]:
                    example = self._convert_image(image_path) # This is an instance of tf.Example
                    writer.write(example.SerializeToString())
            start = end
            end += num_files_per_record
            tfrecord_file_number += 1

    def _convert_image(self, image_path):

        label = get_image_class_from_file_name(image_path)
        image_shape = mpimg.imread(image_path).shape
        if len(image_shape) == 2:
            image_shape += (1,)
        file_name = os.path.basename(image_path)

        # Read a byte representation of the image.
        with tf.gfile.GFile(image_path, 'rb') as fid:
            image_data = fid.read()

        example = tf.train.Example(features = tf.train.Features(feature = {
            'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [file_name.encode('utf-8')])),
            'rows': tf.train.Feature(int64_list = tf.train.Int64List(value = [image_shape[0]])),
            'cols': tf.train.Feature(int64_list = tf.train.Int64List(value = [image_shape[1]])),
            'channels': tf.train.Feature(int64_list = tf.train.Int64List(value = [image_shape[2]])),
            'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data])),
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
        }))

        return example

def get_subdirectories(directory, include_top=True):
    '''
    Return a list of all subdirectories of `directory`.
    '''

    subdirectories = []

    if include_top:
        subdirectories.append(directory)

    for dirpath, dirnames, filenames in os.walk(top=directory, topdown=True):
        subdirs = [os.path.join(dirpath, dirname) for dirname in dirnames]
        subdirectories += subdirs

    return subdirectories

def get_image_paths(directories, extensions=['jpg','jpeg','png']):
    '''
    Return a list of the absolute paths of all images in `directories`.

    Arguments:
        directories (list): A list of directory paths to iterate over.
        extensions (list, optional): An optional list of strings that
            define all acceptable file extensions. If `None`, any file
            extension is acceptable.
    '''

    image_paths = []

    if extensions is None:
        for directory in directories:
            image_paths += glob.glob(os.path.join(directory, '*'))
    else:
        for directory in directories:
            for extension in extensions:
                image_paths += glob.glob(os.path.join(directory, '*.'+extension))

    return image_paths

def get_image_class_from_file_name(image_path, separator='_'):
    '''
    Return the class ID of an image (i.e. an integer) based on the beginning of the
    image name string
    '''

    return int(os.path.basename(image_path).split(separator)[0])
