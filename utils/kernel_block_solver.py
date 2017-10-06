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

"""Block kernel lsqr solver for multi-class classification."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import scipy.linalg as linalg
from scipy.sparse.linalg import spsolve
from sklearn import metrics


class BlockKernelSolver(object):
  """Inspired by algorithm from https://arxiv.org/pdf/1602.05310.pdf."""
  # TODO(lishal): save preformed kernel matrix and reuse if possible
  # perhaps not possible if want to keep scikitlearn signature

  def __init__(self,
               random_state=1,
               C=0.1,
               block_size=4000,
               epochs=3,
               verbose=False,
               gamma=None):
    self.block_size = block_size
    self.epochs = epochs
    self.C = C
    self.kernel = 'rbf'
    self.coef_ = None
    self.verbose = verbose
    self.encode_map = None
    self.decode_map = None
    self.gamma = gamma
    self.X_train = None
    self.random_state = random_state

  def encode_y(self, y):
    # Handles classes that do not start counting from 0.
    if self.encode_map is None:
      self.classes_ = sorted(list(set(y)))
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
    """Form K and solve (K + lambda * I)x = y in a block-wise fashion."""
    np.random.seed(self.random_state)
    self.X_train = X_train
    n_features = X_train.shape[1]
    y = self.encode_y(y_train)
    if self.gamma is None:
      self.gamma = 1./n_features
    K = metrics.pairwise.pairwise_kernels(
        X_train, metric=self.kernel, gamma=self.gamma)
    if self.verbose:
      print('Finished forming kernel matrix.')

    # compute some constants
    num_classes = len(list(set(y)))
    num_samples = K.shape[0]
    num_blocks = math.ceil(num_samples*1.0/self.block_size)
    x = np.zeros((K.shape[0], num_classes))
    y_hat = np.zeros((K.shape[0], num_classes))
    onehot = lambda x: np.eye(num_classes)[x]
    y_onehot = np.array(map(onehot, y))
    idxes = np.diag_indices(num_samples)
    if sample_weight is not None:
      weights = np.sqrt(sample_weight)
      weights = weights[:, np.newaxis]
      y_onehot = weights * y_onehot
      K *= np.outer(weights, weights)
    if num_blocks == 1:
      epochs = 1
    else:
      epochs = self.epochs

    for e in range(epochs):
      shuffled_coords = np.random.choice(
          num_samples, num_samples, replace=False)
      for b in range(int(num_blocks)):
        residuals = y_onehot - y_hat

        # Form a block of K.
        K[idxes] += (self.C * num_samples)
        block = shuffled_coords[b*self.block_size:
                                min((b+1)*self.block_size, num_samples)]
        K_block = K[:, block]
        # Dim should be block size x block size
        KbTKb = K_block.T.dot(K_block)

        if self.verbose:
          print('solving block {0}'.format(b))
        # Try linalg solve then sparse solve for handling of sparse input.
        try:
          x_block = linalg.solve(KbTKb, K_block.T.dot(residuals))
        except:
          try:
            x_block = spsolve(KbTKb, K_block.T.dot(residuals))
          except:
            return None

        # update model
        x[block] = x[block] + x_block
        K[idxes] = K[idxes] - (self.C * num_samples)
        y_hat = K.dot(x)

        y_pred = np.argmax(y_hat, axis=1)
        train_acc = metrics.accuracy_score(y, y_pred)
        if self.verbose:
          print('Epoch: {0}, Block: {1}, Train Accuracy: {2}'
                .format(e, b, train_acc))
    self.coef_ = x

  def predict(self, X_val):
    val_K = metrics.pairwise.pairwise_kernels(
        X_val, self.X_train, metric=self.kernel, gamma=self.gamma)
    val_pred = np.argmax(val_K.dot(self.coef_), axis=1)
    return self.decode_y(val_pred)

  def score(self, X_val, val_y):
    val_pred = self.predict(X_val)
    val_acc = metrics.accuracy_score(val_y, val_pred)
    return val_acc

  def decision_function(self, X, type='predicted'):
    # Return the predicted value of the best class
    # Margin_AL will see that a vector is returned and not a matrix and
    # simply select the points that have the lowest predicted value to label
    K = metrics.pairwise.pairwise_kernels(
        X, self.X_train, metric=self.kernel, gamma=self.gamma)
    predicted = K.dot(self.coef_)
    if type == 'scores':
      val_best = np.max(K.dot(self.coef_), axis=1)
      return val_best
    elif type == 'predicted':
      return predicted
    else:
      raise NotImplementedError('Invalid return type for decision function.')

  def get_params(self, deep=False):
    params = {}
    params['C'] = self.C
    params['gamma'] = self.gamma
    if deep:
      return copy.deepcopy(params)
    return copy.copy(params)

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self

  def softmax_over_predicted(self, X):
    val_K = metrics.pairwise.pairwise_kernels(
        X, self.X_train, metric=self.kernel, gamma=self.gamma)
    val_pred = val_K.dot(self.coef_)
    row_min = np.min(val_pred, axis=1)
    val_pred = val_pred - row_min[:, None]
    val_pred = np.exp(val_pred)
    sum_exp = np.sum(val_pred, axis=1)
    val_pred = val_pred/sum_exp[:, None]
    return val_pred
