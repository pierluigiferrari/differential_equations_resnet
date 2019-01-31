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
import sys
import glob
import random
import pathlib
from tqdm import trange
from math import ceil
from sklearn.model_selection import train_test_split

class TFRecordGenerator:

    def __init__(self):
        pass

    def convert(self,
                input_directory,
                output_directory,
                prefix='',
                suffix='tfrecord',
                num_files_per_record=1000,
                train_val_split=0.25,
                encode_image_shape=False):
        '''
        Convert all images in input_directory and any subdirectories into TFRedord files.

        Along with the images themselves, classification labels will be stored based on the
        file names of the respective images. The images will always be shuffled before being
        written to the TFRecord files.

        Arguments:
            input_directory (str): Images from this directory and any subdirectories will be
                converted to TFRecord files.
            output_directory (str): The directory in which to place the created TFRecord files.
                If it doesn't exist, it will be created. If you are using the built-in feature
                to split the dataset into training and validation parts, then two appropriate
                subdirectories will be created automatically.
            prefix (str, optional): The beginning of the file names for the TFRecord output files.
            suffix (str, optional): The file extension for the TFRecord output files. Must not
                contain a leading dot.
            num_files_per_record (int, optional): The number of images to store in one TFRecord file.
            train_val_split (float, optional): The share of the images that should be split into a
                validation dataset. If `None`, then all images will be associated with the same dataset.
                This is just a convenience feature that ensures the desired split size and names the
                files of the two datasets accordingly.

        Returns:
            None.
        '''

        image_paths = get_image_paths(get_subdirectories(input_directory))

        if not (train_val_split is None):
            train_paths, val_paths = train_test_split(image_paths, test_size=train_val_split, shuffle=True)
            self.train_dataset_size = len(train_paths)
            self.val_dataset_size = len(val_paths)
            print("Number of examples in the training dataset: {}".format(self.train_dataset_size))
            print("Number of examples in the validation dataset: {}".format(self.val_dataset_size))
            self._convert(image_paths=train_paths,
                          output_directory=os.path.join(output_directory, 'train'),
                          prefix=prefix+'_train',
                          suffix=suffix,
                          num_files_per_record=num_files_per_record,
                          encode_image_shape=False)
            self._convert(image_paths=val_paths,
                          output_directory=os.path.join(output_directory, 'val'),
                          prefix=prefix+'_val',
                          suffix=suffix,
                          num_files_per_record=num_files_per_record,
                          encode_image_shape=False)
        else:
            random.shuffle(image_paths)
            self.dataset_size = len(image_paths)
            print("Number of examples in the dataset: {}".format(self.dataset_size))
            self._convert(image_paths=image_paths,
                          output_directory=output_directory,
                          prefix=prefix,
                          suffix=suffix,
                          num_files_per_record=num_files_per_record,
                          encode_image_shape=False)

    def _convert(self,
                 image_paths,
                 output_directory,
                 prefix,
                 suffix,
                 num_files_per_record,
                 encode_image_shape=False):

        pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

        num_files_total = len(image_paths)
        num_tfrecords = ceil(num_files_total / num_files_per_record)
        start = 0
        tfrecord_file_number = 0

        while start < num_files_total:
            num_files_remaining = num_files_total - start
            batch_size = num_files_per_record if (num_files_remaining >= num_files_per_record) else num_files_remaining
            with tf.python_io.TFRecordWriter(os.path.join(output_directory, '') + prefix + '_{:04d}.{}'.format(tfrecord_file_number, suffix)) as writer:
                for i in trange(batch_size, desc="Creating TFRecord file {} {}/{}".format(prefix, tfrecord_file_number+1, num_tfrecords), file=sys.stdout):
                    example = self._convert_sample(image_paths[start+i], encode_image_shape) # This is an instance of tf.Example
                    writer.write(example.SerializeToString())
            start += num_files_per_record
            tfrecord_file_number += 1

    def _convert_sample(self, image_path, encode_image_shape=False):

        # Convert the image.
        image_feature = self._convert_image(image_path, encode_image_shape)

        # Convert the label.
        label_feature = self._convert_image_class_from_file_name(image_path)

        return tf.train.Example(features = tf.train.Features(feature = {**image_feature, **label_feature}))

    def _convert_image(self, image_path, encode_image_shape=False):
        '''
        Converts an image and returns a dictionary of `tf.train.Feature` objects,
        which is the input to `tf.train.Features`.
        '''

        file_name = os.path.basename(image_path)

        # Read a byte representation of the image.
        with tf.gfile.GFile(image_path, 'rb') as fid:
            image = fid.read()

        if encode_image_shape:
            image_shape = mpimg.imread(image_path).shape
            if len(image_shape) == 2:
                image_shape += (1,)
            return {
                'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [file_name.encode('utf-8')])),
                'height': tf.train.Feature(int64_list = tf.train.Int64List(value = [image_shape[0]])),
                'width': tf.train.Feature(int64_list = tf.train.Int64List(value = [image_shape[1]])),
                'channels': tf.train.Feature(int64_list = tf.train.Int64List(value = [image_shape[2]]))
            }

        return {
            'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
            'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [file_name.encode('utf-8')]))
        }

    def _convert_image_class_from_file_name(self, image_path):

        label = get_image_class_from_file_name(image_path)
        return {
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
        }

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
