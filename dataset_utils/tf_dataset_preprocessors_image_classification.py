'''
Various preprocessor classes for image classification to be applied to tf.data.Datasets.

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

class UnpackImagesLabels:
    '''
    Map the elements of a dataset from their original format to a tuple of an image
    and a label tensor.

    Assumes the elements of the input dataset are dictionaries that possess the keys
    'image' and 'label'.
    '''

    def __init__(self, num_parallel_calls=None):
        self.num_parallel_calls = num_parallel_calls

    def __call__(self, dataset):
        dataset = dataset.map(lambda sample: (sample['image'], sample['label']),
                              num_parallel_calls=self.num_parallel_calls)
        return dataset

class ConvertLabelsToOneHot:

    def __init__(self, num_classes, num_parallel_calls=None):
        self.num_classes = num_classes
        self.num_parallel_calls = num_parallel_calls

    def __call__(self, dataset):
        dataset = dataset.map(lambda image, label: (image, tf.one_hot(indices=label,
                                                                      depth=self.num_classes)),
                              num_parallel_calls=self.num_parallel_calls)
        return dataset

class DecodeImages:

    def __init__(self,
                 channels=3,
                 num_parallel_calls=None):

        self.channels = channels
        self.num_parallel_calls = num_parallel_calls

    def __call__(self, dataset):
        dataset = dataset.map(lambda image, label: (tf.image.decode_image(image, channels=self.channels), label),
                              num_parallel_calls=self.num_parallel_calls)
        return dataset

class DecodeJPEGImages:

    def __init__(self,
                 channels=None,
                 ratio=1,
                 num_parallel_calls=None):

        self.channels = channels
        self.ratio = ratio
        self.num_parallel_calls = num_parallel_calls

    def __call__(self, dataset):
        dataset = dataset.map(lambda image, label: (tf.image.decode_jpeg(image,
                                                                         channels=self.channels,
                                                                         ratio=self.ratio), label),
                              num_parallel_calls=self.num_parallel_calls)
        return dataset

class RandomCrop:

    def __init__(self,
                 aspect_ratio=1,
                 scale=0.9,
                 channels=3,
                 num_parallel_calls=None):

        self.aspect_ratio = aspect_ratio
        self.scale = scale
        self.channels = channels
        self.num_parallel_calls = num_parallel_calls

    def __call__(self, dataset):

        def _inner(image, label):

            side_length = tf.to_int32(tf.to_float(tf.minimum(tf.shape(image)[0], tf.shape(image)[1])) * tf.constant(self.scale))

            if self.channels == 1:
                crop_size = [side_length, side_length]
            else:
                crop_size = [side_length, side_length, tf.shape(image)[2]]

            return (tf.image.random_crop(image, size=crop_size), label)

        dataset = dataset.map(_inner,
                              num_parallel_calls=self.num_parallel_calls)
        return dataset

class Resize:

    def __init__(self,
                 target_size,
                 preserve_aspect_ratio=False,
                 num_parallel_calls=None):

        self.target_size = target_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.num_parallel_calls = num_parallel_calls

    def __call__(self, dataset):
        dataset = dataset.map(lambda image, label: (tf.image.resize_images(image,
                                                                           size=self.target_size,
                                                                           preserve_aspect_ratio=self.preserve_aspect_ratio), label),
                              num_parallel_calls=self.num_parallel_calls)
        return dataset

class ResizeWithPad:

    def __init__(self,
                 target_size,
                 num_parallel_calls=None):

        self.target_size = target_size
        self.num_parallel_calls = num_parallel_calls

    def __call__(self, dataset):
        dataset = dataset.map(lambda image, label: (tf.image.resize_image_with_pad(image,
                                                                                   target_height=self.target_size[0],
                                                                                   target_width=self.target_size[1],), label),
                              num_parallel_calls=self.num_parallel_calls)
        return dataset

class RandomFlipLeftRight:

    def __init__(self, num_parallel_calls=None):
        self.num_parallel_calls = num_parallel_calls

    def __call__(self, dataset):
        dataset = dataset.map(lambda image, label: (tf.image.random_flip_left_right(image), label),
                              num_parallel_calls=self.num_parallel_calls)
        return dataset

class RandomBrightness:

    def __init__(self, max_delta=0.5, num_parallel_calls=None):
        self.max_delta = max_delta
        self.num_parallel_calls = num_parallel_calls

    def __call__(self, dataset):
        dataset = dataset.map(lambda image, label: (tf.image.random_brightness(image,
                                                                               max_delta=self.max_delta), label),
                              num_parallel_calls=self.num_parallel_calls)
        return dataset

class RandomSaturation:

    def __init__(self, lower=0.5, upper=1.5, num_parallel_calls=None):
        self.lower = lower
        self.upper = upper
        self.num_parallel_calls = num_parallel_calls

    def __call__(self, dataset):
        dataset = dataset.map(lambda image, label: (tf.image.random_saturation(image,
                                                                               lower=self.lower,
                                                                               upper=self.upper), label),
                              num_parallel_calls=self.num_parallel_calls)
        return dataset
