'''
A training routine for tf.keras models written in the low-level TensorFlow API.

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
from distutils.version import LooseVersion
import warnings
from tqdm import trange
import sys
import os.path
import pathlib
import shutil
from glob import glob
import time
import csv

from training.tf_variable_summaries import add_mean_norm_summary

class Training:

    def __init__(self,
                 model_build_function,
                 optimizer,
                 train_dataset,
                 val_dataset=None,
                 record_summaries=True,
                 summaries=['mean_gradient_norms'],
                 summaries_dir=None,
                 summaries_name=None,
                 csv_logger_dir=None,
                 csv_logger_name=None):
        '''
        Arguments:
            model (function): A function that takes as input an input tensor (the output of a tf.keras.layers.Input object)
                and returns a tf.keras.models.Model object. The returned model must not be compiled.
            optimizer (tf.train.Optimizer object): A tf.train.Optimizer object. The learning rate of the optimizer
                instance is expected to be initialized by a placeholder tensor with the name 'learning_rate'.
            train_dataset (tf.data.Dataset object): A tf.data.Dataset object, the dataset to train on. An appropriate
                iterator for the dataset will be created internally. It is recommended to set the dataset to infinite
                repetition.
            val_dataset (tf.data.Dataset object, optional): A tf.data.Dataset object, the validation dataset.
            record_summaries (bool, optional): Whether or not to record TensorBoard summaries.
            summaries (list, optional): TODO.
            summaries_dir (str, optional): The full path of the directory to which to
                write the summaries protocol buffers.
            summaries_name (str, optional): The name of the summaries buffers.
            csv_logger_dir (str, optional): TODO.
            csv_logger_name (str, optional): TODO.
        '''
        # Check TensorFlow version
        assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'This program requires TensorFlow version 1.0 or newer. You are using {}'.format(tf.__version__)
        print('TensorFlow Version: {}'.format(tf.__version__))

        if not isinstance(optimizer, tf.train.Optimizer):
            raise ValueError("The model you passed is not an instance of tf.train.Optimizer or a subclass thereof.")

        if not isinstance(train_dataset, tf.data.Dataset):
            raise ValueError("train_dataset must be a tf.data.Dataset object.")

        if (not (val_dataset is None)) and (not isinstance(val_dataset, tf.data.Dataset)):
            raise ValueError("val_dataset must be a tf.data.Dataset object.")

        self.model_build_function = model_build_function
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.record_summaries = record_summaries
        self.summaries = summaries
        self.summaries_dir = summaries_dir
        self.summaries_name = summaries_name
        self.csv_logger_dir = csv_logger_dir
        self.csv_logger_name = csv_logger_name

        self.variables_updated = False # Keep track of whether any variable values changed since this model was last saved.
        self.eval_dataset = None # Which dataset to use for evaluation during training. Only relevant for training.

        # The following lists store data about the metrics being tracked.
        # Note that `self.metric_value_tensors` and `self.metric_update_ops` represent
        # the metrics being tracked, not the metrics generally available in the model.
        self.metric_names = [] # Store the metric names here.
        self.metric_values = [] # Store the latest metric evaluations here.
        self.best_metric_values = [] # Keep score of the best historical metric values.
        self.metric_value_tensors = [] # Store the value tensors from tf.metrics here.
        self.metric_update_ops = [] # Store the update ops from tf.metrics here.

        self.training_loss = None
        self.best_training_loss = 99999999.9

        self.sess = tf.Session()
        tf.keras.backend.set_session(self.sess)
        self.g_step = None # The global step

        ########################################################################
        # Build the graph.
        ########################################################################

        # Set up the dataset iterators.
        (self.features,
         self.labels,
         self.iterator,
         self.iterator_handle,
         self.train_iterator_handle,
         self.val_iterator_handle) = self._build_data_iterators()
        # Build the model.
        self.input_tensor = tf.keras.layers.Input(tensor=self.features)
        self.model = self.model_build_function(self.input_tensor)
        if not isinstance(self.model, (tf.keras.models.Model, tf.keras.models.Sequential)):
            raise ValueError("The model_build_function you passed does not return a tf.keras.models.Model or tf.keras.models.Sequential object or a subclass thereof.")
        # Add the loss and optimizer.
        (self.total_loss,
         self.grads_and_vars,
         self.train_step,
         self.learning_rate,
         self.global_step) = self._build_optimizer()
        # Add the prediction outputs.
        self.predictions_argmax = self._build_predictor()
        # Add metrics for evaluation.
        (self.mean_loss_value,
         self.mean_loss_update_op,
         self.acc_value,
         self.acc_update_op,
         self.metrics_reset_op) = self._build_metrics()
        # Add gradient metrics.
        self.gradient_mean_norms = self._build_gradient_metrics()
        self.gradient_mean_norm_names = [tensor.name.split(':')[0].split('metrics/')[1] for tensor in self.gradient_mean_norms]
        # Add summary ops for TensorBoard.
        (self.training_summaries,
         self.evaluation_summaries) = self._build_summary_ops()
        # Initialize the global and local (for the metrics) variables.
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        # Add a saver.
        self.saver = tf.train.Saver(var_list=None,
                                    reshape=False,
                                    max_to_keep=5,
                                    keep_checkpoint_every_n_hours=10000.0)

        ########################################################################
        # Perform other preparatory work.
        ########################################################################

        self._initialize_metrics()

        # Set up the summary file writers.
        if self.record_summaries:
            self.training_writer = tf.summary.FileWriter(logdir=os.path.join(self.summaries_dir, self.summaries_name),
                                                    graph=self.sess.graph)
            self.evaluation_writer = tf.summary.FileWriter(logdir=os.path.join(self.summaries_dir, self.summaries_name + '_eval'))

            # Create the directory for the CSV file if it doesn't already exist.
            pathlib.Path(self.csv_logger_dir).mkdir(parents=True, exist_ok=True)
            # Open existing CSV file or create a new one. The stream is positioned at the end of the file and anything
            # written to the file will always be written at the end of the file, regardless of intermediate 'seek()'
            # calls.
            self.csv_file = open(os.path.join(self.csv_logger_dir,'{}.csv'.format(self.csv_logger_name)), 'a+', newline='')
            # Check whether this file is new (and thus empty) or whether it already has content in it.
            self.csv_file.seek(0)
            is_empty = self.csv_file.readline() == ''
            # Create a CSV writer object.
            self.csv_writer = csv.writer(self.csv_file, delimiter=' ')
            # Write a header row to the file only if it is empty.
            if is_empty:
                self.csv_writer.writerow(['global_step'] + self.gradient_mean_norm_names)

        # Define the fetches and feed dict.
        self.fetches_summaries = {'global_step': self.global_step,
                                  'training_step': self.train_step,
                                  'metric_updates': self.metric_update_ops,
                                  'metric_values': self.metric_value_tensors,
                                  'gradient_mean_norms': self.gradient_mean_norms,
                                  'training_summaries': self.training_summaries,
                                  'layer_updates': self.model.updates}
        self.fetches_no_summaries = {'global_step': self.global_step,
                                     'training_step': self.train_step,
                                     'metric_updates': self.metric_update_ops,
                                     'metric_values': self.metric_value_tensors,
                                     'layer_updates': self.model.updates}

    def _build_data_iterators(self):
        '''
        Set up the iterators for the datasets.
        '''
        with tf.name_scope('data_input'):

            train_iterator = self.train_dataset.make_one_shot_iterator()
            train_iterator_handle = self.sess.run(train_iterator.string_handle())

            if not (self.val_dataset is None):
                val_iterator = self.val_dataset.make_one_shot_iterator()
            else:
                val_iterator = self.train_dataset.make_one_shot_iterator()

            val_iterator_handle = self.sess.run(val_iterator.string_handle())

            iterator_handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(string_handle=iterator_handle,
                                                           output_types=train_iterator.output_types,
                                                           output_shapes=train_iterator.output_shapes)

            features, labels = iterator.get_next()

            return (features,
                    labels,
                    iterator,
                    iterator_handle,
                    train_iterator_handle,
                    val_iterator_handle)

    def _build_optimizer(self):
        '''
        Build the training-relevant part of the graph.
        '''

        with tf.name_scope('optimizer'):
            # Create a training step counter.
            global_step = tf.Variable(0, trainable=False, name='global_step')
            # Get the learning rate placeholder from the optimizer object.
            learning_rate = tf.get_default_graph().get_tensor_by_name('learning_rate:0')
            # Compute the regularizatin loss.
            regularization_losses = self.model.losses # This is a list of the individual loss values, so we still need to sum them up.
            regularization_loss = tf.add_n(regularization_losses, name='regularization_loss') # Scalar
            # Compute the total loss.
            approximation_loss = loss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(target=self.labels, output=self.model.output, from_logits=False), name='approximation_loss') # Scalar
            total_loss = tf.add(approximation_loss, regularization_loss, name='total_loss')
            # Compute the gradients and apply them.
            grads_and_vars = self.optimizer.compute_gradients(loss=total_loss)
            train_step = self.optimizer.apply_gradients(grads_and_vars, global_step=global_step, name='train_op')

        return total_loss, grads_and_vars, train_step, learning_rate, global_step

    def _build_predictor(self):
        '''
        Build the prediction-relevant part of the graph.
        '''

        with tf.name_scope('predictor'):
            predictions_argmax = tf.argmax(self.model.output, axis=-1, name='predictions_argmax', output_type=tf.int64)

        return predictions_argmax

    def _build_metrics(self):
        '''
        Build the evaluation-relevant part of the graph, i.e. the metrics operations.
        '''

        with tf.variable_scope('metrics') as scope:

            labels_argmax = tf.argmax(self.labels, axis=-1, name='labels_argmax', output_type=tf.int64)

            # 1: Mean loss

            mean_loss_value, mean_loss_update_op = tf.metrics.mean(self.total_loss)

            mean_loss_value = tf.identity(mean_loss_value, name='mean_loss_value')
            mean_loss_update_op = tf.identity(mean_loss_update_op, name='mean_loss_update_op')

            # 2: Accuracy

            acc_value, acc_update_op = tf.metrics.accuracy(labels=labels_argmax,
                                                           predictions=self.predictions_argmax)

            acc_value = tf.identity(acc_value, name='acc_value')
            acc_update_op = tf.identity(acc_update_op, name='acc_update_op')

            # As of TensorFlow version 1.3, TensorFlow's streaming metrics don't have reset operations,
            # so we need to create our own as a work-around. Say we want to evaluate
            # a metric after every training epoch. If we didn't have
            # a way to reset the metric's update op after every evaluation,
            # the computed metric value would be the average of the current evaluation
            # and all previous evaluations from past epochs, which is obviously not
            # what we want.
            local_metric_vars = tf.contrib.framework.get_variables(scope=scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            metrics_reset_op = tf.variables_initializer(var_list=local_metric_vars, name='metrics_reset_op')

        return (mean_loss_value,
                mean_loss_update_op,
                acc_value,
                acc_update_op,
                metrics_reset_op)

    def _build_gradient_metrics(self):
        '''
        Build the ops that measure the behavior of the gradients.
        '''

        # The following values are picked so that we get only the gradient norms for the gradients
        # we're interested in, rather than for all gradients.
        min_kernel_rank = 4 # Do we care only about the gradients of the convolutional layers (which have rank 4), or also about those of other layer types, e.g. dense layers?
        min_kernel_size = 3 # Do we only care about convolutions that are at least e.g. 3-by-3, or do we also care about 1-by-1 convolutions?
        valid_variable_types = ['kernel'] # Many models have 'kernels' and 'biases' for the main layers, 'gamma' and 'beta' for BatchNorm, etc.

        gradient_mean_norms = []

        with tf.name_scope('gradient_metrics'):

            for gradient, variable in self.grads_and_vars:

                # Extract the weight type from the variable name.
                variable_name = variable.name.split(':')[0]
                variable_type = variable_name.split('/')[-1]

                if (variable_type in valid_variable_types
                    and variable.shape[0] >= min_kernel_size
                    and len(variable.shape) >= min_kernel_rank):

                    gradient_mean_norms.append(tf.identity(tf.norm(gradient, ord='euclidean') / tf.to_float(tf.size(gradient)), name='{}_gradient_mean_norm'.format(variable_name)))

            return gradient_mean_norms

    def _build_summary_ops(self):
        '''
        Build the part of the graph that logs summaries for TensorBoard.
        '''

        with tf.name_scope('summaries'):

            training_summaries = []

            # Summaries that compute the mean norms of the gradients and variables.
            if 'mean_gradient_norms' in self.summaries:
                for gradient_mean_norm, name in zip(self.gradient_mean_norms, self.gradient_mean_norm_names):
                    training_summaries.append(tf.summary.scalar(name=name, tensor=gradient_mean_norm))

            if 'mean_weight_norms' in self.summaries:
                for gradient, variable in self.grads_and_vars:
                    training_summaries.append(add_mean_norm_summary(variable,
                                                                    scope='{}_gradient'.format(variable.name.replace(':', '_')),
                                                                    order='euclidean'))

            # Summaries for learning rate and metrics.
            training_summaries.append(tf.summary.scalar(name='learning_rate', tensor=self.learning_rate))
            training_summaries.append(tf.summary.scalar(name='mean_loss', tensor=self.mean_loss_value))
            training_summaries.append(tf.summary.scalar(name='accuracy', tensor=self.acc_value))

            tr_summaries = tf.summary.merge(training_summaries, name='training_summaries')

            # All evaluation metrics.
            evaluation_summaries = []

            evaluation_summaries.append(tf.summary.scalar(name='mean_loss', tensor=self.mean_loss_value))
            evaluation_summaries.append(tf.summary.scalar(name='accuracy', tensor=self.acc_value))

            if evaluation_summaries:
                eval_summaries = tf.summary.merge(inputs=evaluation_summaries, name='evaluation_summaries')
                return tr_summaries, eval_summaries
            else:
                return tr_summaries

    def _initialize_metrics(self):
        '''
        Initialize/reset the metrics before every call to `train` and `evaluate`.
        '''

        # Reset lists of previous tracked metrics.
        self.metric_names = []
        self.best_metric_values = []
        self.metric_update_ops = []
        self.metric_value_tensors = []

        # Set the metrics that will be evaluated.
        # Mean loss.
        self.metric_names.append('mean_loss')
        self.best_metric_values.append(99999999.9)
        self.metric_update_ops.append(self.mean_loss_update_op)
        self.metric_value_tensors.append(self.mean_loss_value)
        # Accuracy.
        self.metric_names.append('accuracy')
        self.best_metric_values.append(0.0)
        self.metric_update_ops.append(self.acc_update_op)
        self.metric_value_tensors.append(self.acc_value)

    def train(self,
              epochs,
              steps_per_epoch,
              learning_rate_schedule,
              eval_dataset='train',
              eval_frequency=5,
              eval_steps=None,
              save_during_training=False,
              save_dir=None,
              save_best_only=True,
              save_tags=['default'],
              save_name='',
              save_frequency=5,
              saver='train_saver',
              monitor='loss',
              summaries_frequency=10):
        '''
        Train the model.

        Arguments:
            epochs (int): The number of epochs to run the training for, where each epoch
                consists of `steps_per_epoch` training steps.
            steps_per_epoch (int): The number of training steps (i.e. batches to train on)
                per epoch.
            learning_rate_schedule (function): Any function that takes as its sole input
                an integer (the global step counter) and returns a float (the learning rate).
            eval_dataset (string, optional): Which dataset to use for the evaluation
                of the model during the training. Can be either of 'train' (the train_dataset
                will be used) or 'val' (the val_dataset will be used). Defaults to 'train',
                but should be set to 'val' if a validation dataset is available.
            eval_frequency (int, optional): The model will be evaluated on the evaluation
                dataset after every `eval_frequency` epochs. If `None`, the model will not be
                evaluated in between epochs.
            eval_steps (int, optional): The number of iterations to run over the evaluation
                dataset during evaluation. If this is `None`, it defaults to `steps_per_epoch`.
            save_during_training (bool, optional): Whether or not to save the model periodically
                during training, the parameters of which can be set in the subsequent arguments.
            save_dir (string, optional): The full path of the directory to save the model to
                during training.
            save_best_only (bool, optional): If `True`, the model will only be saved upon
                evaluation if the metric defined by `monitor` has improved since it was last
                measured before. Can only be `True` if `metrics` is not empty.
            save_tags (list, optional): An optional list of tags to save the model metagraph
                with in the SavedModel protocol buffer. At least one tag must be given.
            save_name (string, optional): An optional name string to include in the name of
                the folder in which the model will be saved during training. Note that what
                you pass as the name here will be only part of the folder name. The folder
                name also includes a count of the global training step and the values of
                any metrics that are being evaluate, although at least the training loss.
                It is hence not necessary to pass a name here, each saved model will be
                uniquely and descriptively named regardless.
            save_frequency (int, optional): The model will be saved at most after every
                `save_frequency` epochs, but possibly less often if `save_best_only` is `True`
                and if there was no improvement in the monitored metric.
            saver (string, optional): Which saver to use when saving the model during training.
                Can be either of 'saved_model' in order to use `tf.saved_model` or 'train_saver'
                in order to use `tf.train.Saver`. Check the TensorFlow documentation for details
                on which saver might be better for your use case. In general you can't go wrong
                with either of the two.
            monitor (string, optional): The name of the metric that is to be monitored in
                order to decide whether the model should be saved. Can be one of
                `{'loss', 'accuracy'}`, which are the currently available metrics.
            summaries_frequency (int, optional): How often summaries should be logged for
                tensors which are updated at every training step. The summaries for such tensors
                will be recorded every `summaries_frequency` training steps.
        '''

        # Check for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found. Please note that training this network will be unbearably slow without a GPU.')
        else:
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

        if not eval_dataset in ['train', 'val']:
            raise ValueError("`eval_dataset` must be one of 'train' or 'val', but is '{}'.".format(eval_dataset))

        if (eval_dataset == 'val') and ((self.val_dataset is None) or (eval_steps is None)):
            raise ValueError("When eval_dataset == 'val', a `val_dataset` and `val_steps` must be passed.")

        self._initialize_metrics()

        if monitor == 'loss': monitor = 'mean_loss'
        if (not monitor in self.metric_names):
            raise ValueError('You are trying to monitor {}, which is not an available metric.'.format(monitor))

        if eval_steps is None:
            eval_steps = steps_per_epoch

        self.eval_dataset = eval_dataset

        self.g_step = self.sess.run(self.global_step)
        learning_rate = learning_rate_schedule(self.g_step)

        for epoch in range(1, epochs+1):

            ####################################################################
            # Run the training for this epoch.
            ####################################################################

            tr = trange(steps_per_epoch, file=sys.stdout)
            tr.set_description('Epoch {}/{}'.format(epoch, epochs))

            # Reset all stateful metrics before this epoch begins.
            self.sess.run(self.metrics_reset_op)

            for train_step in tr:

                if self.record_summaries and (self.g_step % summaries_frequency == 0):
                    ret = self.sess.run(fetches=self.fetches_summaries, feed_dict={self.iterator_handle: self.train_iterator_handle,
                                                                                   self.learning_rate: learning_rate,
                                                                                   tf.keras.backend.learning_phase(): 1})
                    self.training_writer.add_summary(summary=ret['training_summaries'], global_step=self.g_step)
                    self.csv_writer.writerow([self.g_step - 1] + ret['gradient_mean_norms'])
                else:
                    ret = self.sess.run(fetches=self.fetches_no_summaries, feed_dict={self.iterator_handle: self.train_iterator_handle,
                                                                                      self.learning_rate: learning_rate,
                                                                                      tf.keras.backend.learning_phase(): 1})
                self.g_step = ret['global_step']
                self.variables_updated = True
                self.training_loss = ret['metric_values'][0]

                tr.set_postfix(ordered_dict=dict(zip(self.metric_names + ['global_step'], ret['metric_values'] + [self.g_step])))

                learning_rate = learning_rate_schedule(self.g_step)

            ####################################################################
            # Maybe evaluate the model after this epoch.
            ####################################################################

            if (not (eval_frequency is None )) and (epoch % eval_frequency == 0):

                if eval_dataset == 'train':
                    description = 'Evaluation on training dataset'
                elif eval_dataset == 'val':
                    description = 'Evaluation on validation dataset'

                self._evaluate(eval_dataset=eval_dataset,
                               num_batches=eval_steps,
                               description=description)

                if self.record_summaries:
                    evaluation_summary = self.sess.run(self.evaluation_summaries)
                    self.evaluation_writer.add_summary(summary=evaluation_summary, global_step=self.g_step)

            ####################################################################
            # Maybe save the model after this epoch.
            ####################################################################

            if save_during_training and (epoch % save_frequency == 0):

                save = False
                if save_best_only:
                    if (monitor == 'loss' and
                        (not 'loss' in self.metric_names) and
                        self.training_loss < self.best_training_loss):
                        save = True
                    else:
                        i = self.metric_names.index(monitor)
                        if (monitor == 'loss') and (self.metric_values[i] < self.best_metric_values[i]):
                            save = True
                        elif (monitor in ['accuracry']) and (self.metric_values[i] > self.best_metric_values[i]):
                            save = True
                    if save:
                        print('New best {} value, saving model.'.format(monitor))
                    else:
                        print('No improvement over previous best {} value, not saving model.'.format(monitor))
                else:
                    save = True

                if save:
                    self.save(model_save_dir=save_dir,
                              saver=saver,
                              tags=save_tags,
                              name=save_name,
                              include_global_step=True,
                              include_last_training_loss=True,
                              include_metrics=(len(self.metric_names) > 0))

            ####################################################################
            # Update the current best metric values.
            ####################################################################

            if self.training_loss < self.best_training_loss:
                self.best_training_loss = self.training_loss

            if epoch % eval_frequency == 0:

                for i, metric_name in enumerate(self.metric_names):
                    if (metric_name == 'loss') and (self.metric_values[i] < self.best_metric_values[i]):
                        self.best_metric_values[i] = self.metric_values[i]
                    elif (metric_name in ['accuracry']) and (self.metric_values[i] > self.best_metric_values[i]):
                        self.best_metric_values[i] = self.metric_values[i]

    def _evaluate(self, eval_dataset, num_batches, description='Running evaluation'):
        '''
        Internal method used by both `evaluate()` and `train()` that performs
        the actual evaluation. For the first three arguments, please refer
        to the documentation of the public `evaluate()` method.

        Arguments:
            description (string, optional): A description string that will be prepended
                to the progress bar while the evaluation is being processed. During
                training, this description is used to clarify whether the evaluation
                is being performed on the training or validation dataset.
        '''

        # Reset all metrics' accumulator variables.
        self.sess.run(self.metrics_reset_op)

        # Set up the progress bar.
        tr = trange(num_batches, file=sys.stdout)
        tr.set_description(description)

        # Set the correct iterator handle for the evaluation dataset.
        if eval_dataset == 'val':
            feed_dict = {self.iterator_handle: self.val_iterator_handle,
                         tf.keras.backend.learning_phase(): 0}
        else:
            feed_dict = {self.iterator_handle: self.train_iterator_handle,
                         tf.keras.backend.learning_phase(): 0}

        # Accumulate metrics in batches.
        for step in tr:

            ret = self.sess.run(fetches={'metric_values': self.metric_value_tensors,
                                         'metric_update_ops': self.metric_update_ops}, feed_dict=feed_dict)

            tr.set_postfix(ordered_dict=dict(zip(self.metric_names, ret['metric_values'])))

        self.metric_values = ret['metric_values']

    def evaluate(self, eval_dataset, num_batches, dataset='val'):
        '''
        Evaluates the model on the given metrics on the data in `dataset`.

        Arguments:
            dataset TODO (tf.data.Dataset object): A dataset that produces batches of images
                and associated ground truth images in two separate Numpy arrays.
                The images must be a 4D array with format `(batch_size, height, width, channels)`
                and the ground truth images must be a 4D array with format
                `(batch_size, height, width, num_classes)`, i.e. the ground truth
                data must be provided in one-hot format. The dataset's batch size
                has no effect on the outcome of the evaluation.
            num_batches (int): The number of batches to evaluate the model on.
                Typically this will be the number of batches such that the model
                is being evaluated on the whole evaluation dataset.
            metrics (set, optional): The metrics to be evaluated. A Python set containing
                any subset of `{'loss', 'accuracy'}`, which are the
                currently available metrics.
            dataset (string, optional): Specifies the kind of dataset on which the model
                is being evaluated. Should be set to 'train' if the model is being evaluated
                on a dataset on which it has also been trained, or 'val' if the model is
                being evaluated on a dataset which it has never seen during training.
                This argument has no effect on the evaluation of the model, but if you
                save the model using `save()` after evaluating it, the model name will
                include this value to indicate whether or not the metric values were
                achieved on a dataset that has not been used during training.
        '''

        for metric in metrics:
            if not metric in ['loss', 'accuracy']:
                raise ValueError("{} is not a valid metric. Valid metrics are ['loss', 'accuracy']".format(metric))

        if not eval_dataset in {'train', 'val'}:
            raise ValueError("`dataset` must be either 'train' or 'val'.")

        self._initialize_metrics()

        self._evaluate(eval_dataset, num_batches, description='Running evaluation')

        if eval_dataset == 'val':
            self.eval_dataset = 'val'
        else:
            self.eval_dataset = 'train'

    def predict(self, images, argmax=True):
        '''
        Makes predictions for the input images.

        Arguments:
            images (array-like): The input image or images. Must be an array-like
                object of rank 4. If predicting only one image, encapsulate it in
                a Python list.
            argmax (bool, optional): If `True`, the model predicts class IDs,
                i.e. the last dimension has length 1 and an integer between
                zero and `num_classes - 1` for each pixel. Otherwise, the model
                outputs the softmax distribution, i.e. the last dimension has
                length `num_classes` and contains the probability for each class
                for all pixels.

        Returns:
            The prediction, an array of rank 4 of which the first three dimensions
            are identical to the input and the fourth dimension is as described
            in `argmax`.
        '''
        if argmax:
            return self.sess.run(self.predictions_argmax,
                                 feed_dict={self.image_input: images,
                                            self.keep_prob: 1.0})
        else:
            return self.sess.run(self.softmax_output,
                                 feed_dict={self.image_input: images,
                                            self.keep_prob: 1.0})

    def save(self,
             model_save_dir,
             saver,
             tags=['default'],
             name=None,
             include_global_step=True,
             include_last_training_loss=True,
             include_metrics=True,
             force_save=False):
        '''
        Saves the model to disk.

        Arguments:
            model_save_dir (string): The full path of the directory to which to
                save the model.
            saver (string, optional): Which saver to use when saving the model during training.
                Can be either of 'saved_model' in order to use `tf.saved_model` or 'train_saver'
                in order to use `tf.train.Saver`. Check the TensorFlow documentation for details
                on which saver might be better for your use case. In general you can't go wrong
                with either of the two.
            tags (list, optional): An optional list of tags to save the model metagraph
                with in the SavedModel protocol buffer. At least one tag must be given.
            name (string, optional): An optional name that will be part of the name of the
                saved model's parent directory. Since you have the possibility to include
                the global step number and the values of metrics in the model name, giving
                an explicit name here is often not necessary.
            include_global_step (bool, optional): Whether or not to include the global
                step number in the model name.
            include_last_training_loss (bool, optional): Whether of not to include the
                last training loss value in the model name.
            include_metrics (bool, optional): If `True`, the last values of all recorded
                metrics will be included in the model name.
            force_save (bool, optional): If `True`, force the saver to save the model
                even if no variables have changed since saving last.
        '''

        if (not self.variables_updated) and (not force_save):
            print("Abort: Nothing to save, no training has been performed since the model was last saved.")
            return

        if not saver in {'saved_model', 'train_saver'}:
            raise ValueError("Unexpected value for `saver`: Can be either 'saved_model' or 'train_saver', but received '{}'.".format(saver))

        # Create the save directory if it doesn't already exist.
        pathlib.Path(model_save_dir).mkdir(parents=True, exist_ok=True)

        if self.training_loss is None:
            include_last_training_loss = False

        model_name = 'saved_model'
        if not name is None:
            model_name += '_' + name
        if include_global_step:
            self.g_step = self.sess.run(self.global_step)
            model_name += '_(globalstep-{})'.format(self.g_step)
        if include_last_training_loss:
            model_name += '_(trainloss-{:.4f})'.format(self.training_loss)
        if include_metrics:
            if self.eval_dataset == 'val':
                model_name += '_(eval_on_val_dataset)'
            else:
                model_name += '_(eval_on_train_dataset)'
            for i in range(len(self.metric_names)):
                model_name += '_({}-{:.4f})'.format(self.metric_names[i], self.metric_values[i])
        if not (include_global_step or include_last_training_loss or include_metrics) and (name is None):
            model_name += '_{}'.format(time.time())

        if saver == 'saved_model':
            saved_model_builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(model_save_dir, model_name))
            saved_model_builder.add_meta_graph_and_variables(sess=self.sess, tags=tags)
            saved_model_builder.save()
        else:
            self.saver.save(self.sess,
                            save_path=os.path.join(model_save_dir, model_name, 'variables'),
                            write_meta_graph=True,
                            write_state=True)

        self.variables_updated = False

    def load_variables(self, path):
        '''
        Load variable values into the current model. Only works for variables that
        were saved with 'train_saver'. See `save()` for details.
        '''
        self.saver.restore(self.sess, path)

    def close(self):
        '''
        Closes the session. This method is important to call when you are done working
        with the model in order to release the resources it occupies.
        '''
        self.sess.close()
        if self.record_summaries:
            self.csv_file.close()
        print("The session has been closed.")
