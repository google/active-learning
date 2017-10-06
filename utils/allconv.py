# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements allconv model in keras using tensorflow backend."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import keras
import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential

import numpy as np
import tensorflow as tf


class AllConv(object):
  """allconv network that matches sklearn api."""

  def __init__(self,
               random_state=1,
               epochs=50,
               batch_size=32,
               solver='rmsprop',
               learning_rate=0.001,
               lr_decay=0.):
    # params
    self.solver = solver
    self.epochs = epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.lr_decay = lr_decay
    # data
    self.encode_map = None
    self.decode_map = None
    self.model = None
    self.random_state = random_state
    self.n_classes = None

  def build_model(self, X):
    # assumes that data axis order is same as the backend
    input_shape = X.shape[1:]
    np.random.seed(self.random_state)
    tf.set_random_seed(self.random_state)

    model = Sequential()
    model.add(Conv2D(96, (3, 3), padding='same',
                     input_shape=input_shape, name='conv1'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), name='conv2', padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), strides=(2, 2), padding='same', name='conv3'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), name='conv4', padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), name='conv5', padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), strides=(2, 2), name='conv6', padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), name='conv7', padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), name='conv8', padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1),  name='conv9', padding='valid'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax', name='activation_top'))
    model.summary()

    try:
      optimizer = getattr(keras.optimizers, self.solver)
    except:
      raise NotImplementedError('optimizer not implemented in keras')
    # All optimizers with the exception of nadam take decay as named arg
    try:
      opt = optimizer(lr=self.learning_rate, decay=self.lr_decay)
    except:
      opt = optimizer(lr=self.learning_rate, schedule_decay=self.lr_decay)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    # Save initial weights so that model can be retrained with same
    # initialization
    self.initial_weights = copy.deepcopy(model.get_weights())

    self.model = model

  def create_y_mat(self, y):
    y_encode = self.encode_y(y)
    y_encode = np.reshape(y_encode, (len(y_encode), 1))
    y_mat = keras.utils.to_categorical(y_encode, self.n_classes)
    return y_mat

  # Add handling for classes that do not start counting from 0
  def encode_y(self, y):
    if self.encode_map is None:
      self.classes_ = sorted(list(set(y)))
      self.n_classes = len(self.classes_)
      self.encode_map = dict(zip(self.classes_, range(len(self.classes_))))
      self.decode_map = dict(zip(range(len(self.classes_)), self.classes_))
    mapper = lambda x: self.encode_map[x]
    transformed_y = np.array(map(mapper, y))
    return transformed_y

  def decode_y(self, y):
    mapper = lambda x: self.decode_map[x]
    transformed_y = np.array(map(mapper, y))
    return transformed_y

  def fit(self, X_train, y_train, sample_weight=None):
    y_mat = self.create_y_mat(y_train)

    if self.model is None:
      self.build_model(X_train)

    # We don't want incremental fit so reset learning rate and weights
    K.set_value(self.model.optimizer.lr, self.learning_rate)
    self.model.set_weights(self.initial_weights)
    self.model.fit(
        X_train,
        y_mat,
        batch_size=self.batch_size,
        epochs=self.epochs,
        shuffle=True,
        sample_weight=sample_weight,
        verbose=0)

  def predict(self, X_val):
    predicted = self.model.predict(X_val)
    return predicted

  def score(self, X_val, val_y):
    y_mat = self.create_y_mat(val_y)
    val_acc = self.model.evaluate(X_val, y_mat)[1]
    return val_acc

  def decision_function(self, X):
    return self.predict(X)

  def transform(self, X):
    model = self.model
    inp = [model.input]
    activations = []

    # Get activations of the last conv layer.
    output = [layer.output for layer in model.layers if
              layer.name == 'conv9'][0]
    func = K.function(inp + [K.learning_phase()], [output])
    for i in range(int(X.shape[0]/self.batch_size) + 1):
      minibatch = X[i * self.batch_size
                    : min(X.shape[0], (i+1) * self.batch_size)]
      list_inputs = [minibatch, 0.]
      # Learning phase. 0 = Test mode (no dropout or batch normalization)
      layer_output = func(list_inputs)[0]
      activations.append(layer_output)
    output = np.vstack(tuple(activations))
    output = np.reshape(output, (output.shape[0],np.product(output.shape[1:])))
    return output

  def get_params(self, deep = False):
    params = {}
    params['solver'] = self.solver
    params['epochs'] = self.epochs
    params['batch_size'] = self.batch_size
    params['learning_rate'] = self.learning_rate
    params['weight_decay'] = self.lr_decay
    if deep:
      return copy.deepcopy(params)
    return copy.copy(params)

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self
