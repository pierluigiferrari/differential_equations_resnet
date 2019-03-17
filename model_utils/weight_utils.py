'''
Utilities to save and load weights from and into tf.keras models in certain ways.

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

def pickle_model_weights(model, save_filename):
    '''
    Saves a pickled version of a tf.keras model's weights. Works only for layer types
    which have exactly two weights objects: A kernel and a bias, in this order. The pickled
    file contains a list of layers for only those layers that have trainable weights. Each
    list entry contains a dictionary with the keys 'kernel' and 'bias'.
    '''

    weights = []

    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            kernel, bias = layer.get_weights()
            weights.append({'kernel': kernel, 'bias': bias})

    with open(save_filename, 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)

def double_load_weights(model, weights_pickle_file):
    '''
    Loads weights of a single-block ResNet with (l+2) layers into a single-block
    ResNet with (2l+2) layers, where the weights of the l layers of the saved model
    are each loaded twice into two consecutive layers of the (2l+2)-layer model.
    This is a means of scaling up the layer depth of a model by doubling the number
    of identical layers consecutively.

    A single-block ResNet consists of an initial convolution layer, followed by
    l ResNet blocks, followed by one final dense layer, hence the designation "l+2".
    '''

    # Get the weights from the saved model.
    with open(weights_pickle_file, 'rb') as f:
        saved_model_trainable_layers = pickle.load(f)

    # Get all the layers with weights from the new model.
    new_model_trainable_layers = []
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            new_model_trainable_layers.append(layer)

    #iterator = iter(range(len(saved_model_trainable_layers)))
    # Stage 1 conv layer.
    kernel = saved_model_trainable_layers[0]['kernel']
    bias = saved_model_trainable_layers[0]['bias']
    new_model_trainable_layers[0].set_weights([kernel, bias])

    # All other stages.
    for l in range(1, len(saved_model_trainable_layers)-1):
        kernel = saved_model_trainable_layers[l]['kernel']
        bias = saved_model_trainable_layers[l]['bias']
        new_model_trainable_layers[2*(l-1)+1].set_weights([kernel, bias])
        new_model_trainable_layers[2*l].set_weights([kernel, bias])

    # Final dense layer.
    kernel = saved_model_trainable_layers[-1]['kernel']
    bias = saved_model_trainable_layers[-1]['bias']
    new_model_trainable_layers[-1].set_weights([kernel, bias])
