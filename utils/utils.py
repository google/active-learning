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

"""Utility functions for run_experiment.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import pickle
import sys

import numpy as np
import scipy

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from tensorflow import gfile


from utils.kernel_block_solver import BlockKernelSolver
from utils.small_cnn import SmallCNN
from utils.allconv import AllConv


class Logger(object):
  """Logging object to write to file and stdout."""

  def __init__(self, filename):
    self.terminal = sys.stdout
    self.log = gfile.GFile(filename, "w")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)

  def flush(self):
    self.terminal.flush()

  def flush_file(self):
    self.log.flush()


def create_checker_unbalanced(split, n, grid_size):
  """Creates a dataset with two classes that occupy one color of checkboard.

  Args:
    split: splits to use for class imbalance.
    n: number of datapoints to sample.
    grid_size: checkerboard size.
  Returns:
    X: 2d features.
    y: binary class.
  """
  y = np.zeros(0)
  X = np.zeros((0, 2))
  for i in range(grid_size):
    for j in range(grid_size):
      label = 0
      n_0 = int(n/(grid_size*grid_size) * split[0] * 2)
      if (i-j) % 2 == 0:
        label = 1
        n_0 = int(n/(grid_size*grid_size) * split[1] * 2)
      x_1 = np.random.uniform(i, i+1, n_0)
      x_2 = np.random.uniform(j, j+1, n_0)
      x = np.vstack((x_1, x_2))
      x = x.T
      X = np.concatenate((X, x))
      y_0 = label * np.ones(n_0)
      y = np.concatenate((y, y_0))
  return X, y


def flatten_X(X):
  shape = X.shape
  flat_X = X
  if len(shape) > 2:
    flat_X = np.reshape(X, (shape[0], np.product(shape[1:])))
  return flat_X


def get_mldata(data_dir, name):
  """Loads data from data_dir.

  Looks for the file in data_dir.
  Assumes that data is in pickle format with dictionary fields data and target.


  Args:
    data_dir: directory to look in
    name: dataset name, assumes data is saved in the save_dir with filename
      <name>.pkl
  Returns:
    data and targets
  Raises:
    NameError: dataset not found in data folder.
  """
  dataname = name
  if dataname == "checkerboard":
    X, y = create_checker_unbalanced(split=[1./5, 4./5], n=10000, grid_size=4)
  else:
    filename = os.path.join(data_dir, dataname + ".pkl")
    if not gfile.Exists(filename):
      raise NameError("ERROR: dataset not available")
    data = pickle.load(gfile.GFile(filename, "r"))
    X = data["data"]
    y = data["target"]
    if "keras" in dataname:
      X = X / 255
      y = y.flatten()
  return X, y


def filter_data(X, y, keep=None):
  """Filters data by class indicated in keep.

  Args:
    X: train data
    y: train targets
    keep: defaults to None which will keep everything, otherwise takes a list
      of classes to keep

  Returns:
    filtered data and targets
  """
  if keep is None:
    return X, y
  keep_ind = [i for i in range(len(y)) if y[i] in keep]
  return X[keep_ind], y[keep_ind]


def get_class_counts(y_full, y):
  """Gets the count of all classes in a sample.

  Args:
    y_full: full target vector containing all classes
    y: sample vector for which to perform the count
  Returns:
    count of classes for the sample vector y, the class order for count will
    be the same as long as same y_full is fed in
  """
  classes = np.unique(y_full)
  classes = np.sort(classes)
  unique, counts = np.unique(y, return_counts=True)
  complete_counts = []
  for c in classes:
    if c not in unique:
      complete_counts.append(0)
    else:
      index = np.where(unique == c)[0][0]
      complete_counts.append(counts[index])
  return np.array(complete_counts)


def flip_label(y, percent_random):
  """Flips a percentage of labels for one class to the other.

  Randomly sample a percent of points and randomly label the sampled points as
  one of the other classes.
  Does not introduce bias.

  Args:
    y: labels of all datapoints
    percent_random: percent of datapoints to corrupt the labels

  Returns:
    new labels with noisy labels for indicated percent of data
  """
  classes = np.unique(y)
  y_orig = copy.copy(y)
  indices = range(y_orig.shape[0])
  np.random.shuffle(indices)
  sample = indices[0:int(len(indices) * 1.0 * percent_random)]
  fake_labels = []
  for s in sample:
    label = y[s]
    class_ind = np.where(classes == label)[0][0]
    other_classes = np.delete(classes, class_ind)
    np.random.shuffle(other_classes)
    fake_label = other_classes[0]
    assert fake_label != label
    fake_labels.append(fake_label)
  y[sample] = np.array(fake_labels)
  assert all(y[indices[len(sample):]] == y_orig[indices[len(sample):]])
  return y


def get_model(method, seed=13):
  """Construct sklearn model using either logistic regression or linear svm.

  Wraps grid search on regularization parameter over either logistic regression
  or svm, returns constructed model

  Args:
    method: string indicating scikit method to use, currently accepts logistic
      and linear svm.
    seed: int or rng to use for random state fed to scikit method

  Returns:
    scikit learn model
  """
  # TODO(lishal): extend to include any scikit model that implements
  #   a decision function.
  # TODO(lishal): for kernel methods, currently using default value for gamma
  # but should probably tune.
  if method == "logistic":
    model = LogisticRegression(random_state=seed, multi_class="multinomial",
                               solver="lbfgs", max_iter=200)
    params = {"C": [10.0**(i) for i in range(-4, 5)]}
  elif method == "logistic_ovr":
    model = LogisticRegression(random_state=seed)
    params = {"C": [10.0**(i) for i in range(-5, 4)]}
  elif method == "linear_svm":
    model = LinearSVC(random_state=seed)
    params = {"C": [10.0**(i) for i in range(-4, 5)]}
  elif method == "kernel_svm":
    model = SVC(random_state=seed)
    params = {"C": [10.0**(i) for i in range(-4, 5)]}
  elif method == "kernel_ls":
    model = BlockKernelSolver(random_state=seed)
    params = {"C": [10.0**(i) for i in range(-6, 1)]}
  elif method == "small_cnn":
    # Model does not work with weighted_expert or simulate_batch
    model = SmallCNN(random_state=seed)
    return model
  elif method == "allconv":
    # Model does not work with weighted_expert or simulate_batch
    model = AllConv(random_state=seed)
    return model

  else:
    raise NotImplementedError("ERROR: " + method + " not implemented")

  model = GridSearchCV(model, params, cv=3)
  return model


def calculate_entropy(batch_size, y_s):
  """Calculates KL div between training targets and targets selected by AL.

  Args:
    batch_size: batch size of datapoints selected by AL
    y_s: vector of datapoints selected by AL.  Assumes that the order of the
      data is the order in which points were labeled by AL.  Also assumes
      that in the offline setting y_s will eventually overlap completely with
      original training targets.
  Returns:
    entropy between actual distribution of classes and distribution of
    samples selected by AL
  """
  n_batches = int(np.ceil(len(y_s) * 1.0 / batch_size))
  counts = get_class_counts(y_s, y_s)
  true_dist = counts / (len(y_s) * 1.0)
  entropy = []
  for b in range(n_batches):
    sample = y_s[b * batch_size:(b + 1) * batch_size]
    counts = get_class_counts(y_s, sample)
    sample_dist = counts / (1.0 * len(sample))
    entropy.append(scipy.stats.entropy(true_dist, sample_dist))
  return entropy


def get_train_val_test_splits(X, y, max_points, seed, confusion, seed_batch,
                              split=(2./3, 1./6, 1./6)):
  """Return training, validation, and test splits for X and y.

  Args:
    X: features
    y: targets
    max_points: # of points to use when creating splits.
    seed: seed for shuffling.
    confusion: labeling noise to introduce.  0.1 means randomize 10% of labels.
    seed_batch: # of initial datapoints to ensure sufficient class membership.
    split: percent splits for train, val, and test.
  Returns:
    indices: shuffled indices to recreate splits given original input data X.
    y_noise: y with noise injected, needed to reproduce results outside of
      run_experiments using original data.
  """
  np.random.seed(seed)
  X_copy = copy.copy(X)
  y_copy = copy.copy(y)

  # Introduce labeling noise
  y_noise = flip_label(y_copy, confusion)

  indices = np.arange(len(y))

  if max_points is None:
    max_points = len(y_noise)
  else:
    max_points = min(len(y_noise), max_points)
  train_split = int(max_points * split[0])
  val_split = train_split + int(max_points * split[1])
  assert seed_batch <= train_split

  # Do this to make sure that the initial batch has examples from all classes
  min_shuffle = 3
  n_shuffle = 0
  y_tmp = y_noise

  # Need at least 4 obs of each class for 2 fold CV to work in grid search step
  while (any(get_class_counts(y_tmp, y_tmp[0:seed_batch]) < 4)
         or n_shuffle < min_shuffle):
    np.random.shuffle(indices)
    y_tmp = y_noise[indices]
    n_shuffle += 1

  X_train = X_copy[indices[0:train_split]]
  X_val = X_copy[indices[train_split:val_split]]
  X_test = X_copy[indices[val_split:max_points]]
  y_train = y_noise[indices[0:train_split]]
  y_val = y_noise[indices[train_split:val_split]]
  y_test = y_noise[indices[val_split:max_points]]
  # Make sure that we have enough observations of each class for 2-fold cv
  assert all(get_class_counts(y_noise, y_train[0:seed_batch]) >= 4)
  # Make sure that returned shuffled indices are correct
  assert all(y_noise[indices[0:max_points]] ==
             np.concatenate((y_train, y_val, y_test), axis=0))
  return (indices[0:max_points], X_train, y_train,
          X_val, y_val, X_test, y_test, y_noise)
