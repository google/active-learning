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

"""Make datasets and save specified directory.

Downloads datasets using scikit datasets and can also parse csv file
to save into pickle format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO
import os
import pickle
import StringIO
import tarfile
import urllib2

import keras.backend as K
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
import sklearn.datasets.rcv1
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from google.apputils import app
import gflags as flags
from tensorflow import gfile

flags.DEFINE_string('save_dir', '/tmp/data',
                    'Where to save outputs')
flags.DEFINE_string('datasets', '',
                    'Which datasets to download, comma separated.')
FLAGS = flags.FLAGS


class Dataset(object):

  def __init__(self, X, y):
    self.data = X
    self.target = y


def get_csv_data(filename):
  """Parse csv and return Dataset object with data and targets.

  Create pickle data from csv, assumes the first column contains the targets
  Args:
    filename: complete path of the csv file
  Returns:
    Dataset object
  """
  f = gfile.GFile(filename, 'r')
  mat = []
  for l in f:
    row = l.strip()
    row = row.replace('"', '')
    row = row.split(',')
    row = [float(x) for x in row]
    mat.append(row)
  mat = np.array(mat)
  y = mat[:, 0]
  X = mat[:, 1:]
  data = Dataset(X, y)
  return data


def get_wikipedia_talk_data():
  """Get wikipedia talk dataset.

  See here for more information about the dataset:
  https://figshare.com/articles/Wikipedia_Detox_Data/4054689
  Downloads annotated comments and annotations.
  """

  ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7554634'
  ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7554637'

  def download_file(url):
    req = urllib2.Request(url)
    response = urllib2.urlopen(req)
    return response

  # Process comments
  comments = pd.read_table(
      download_file(ANNOTATED_COMMENTS_URL), index_col=0, sep='\t')
  # remove newline and tab tokens
  comments['comment'] = comments['comment'].apply(
      lambda x: x.replace('NEWLINE_TOKEN', ' '))
  comments['comment'] = comments['comment'].apply(
      lambda x: x.replace('TAB_TOKEN', ' '))

  # Process labels
  annotations = pd.read_table(download_file(ANNOTATIONS_URL), sep='\t')
  # labels a comment as an atack if the majority of annoatators did so
  labels = annotations.groupby('rev_id')['attack'].mean() > 0.5

  # Perform data preprocessing, should probably tune these hyperparameters
  vect = CountVectorizer(max_features=30000, ngram_range=(1, 2))
  tfidf = TfidfTransformer(norm='l2')
  X = tfidf.fit_transform(vect.fit_transform(comments['comment']))
  y = np.array(labels)
  data = Dataset(X, y)
  return data


def get_keras_data(dataname):
  """Get datasets using keras API and return as a Dataset object."""
  if dataname == 'cifar10_keras':
    train, test = cifar10.load_data()
  elif dataname == 'cifar100_coarse_keras':
    train, test = cifar100.load_data('coarse')
  elif dataname == 'cifar100_keras':
    train, test = cifar100.load_data()
  elif dataname == 'mnist_keras':
    train, test = mnist.load_data()
  else:
    raise NotImplementedError('dataset not supported')

  X = np.concatenate((train[0], test[0]))
  y = np.concatenate((train[1], test[1]))

  if dataname == 'mnist_keras':
    # Add extra dimension for channel
    num_rows = X.shape[1]
    num_cols = X.shape[2]
    X = X.reshape(X.shape[0], 1, num_rows, num_cols)
    if K.image_data_format() == 'channels_last':
      X = X.transpose(0, 2, 3, 1)

  y = y.flatten()
  data = Dataset(X, y)
  return data


# TODO(lishal): remove regular cifar10 dataset and only use dataset downloaded
# from keras to maintain image dims to create tensor for tf models
# Requires adding handling in run_experiment.py for handling of different
# training methods that require either 2d or tensor data.
def get_cifar10():
  """Get CIFAR-10 dataset from source dir.

  Slightly redundant with keras function to get cifar10 but this returns
  in flat format instead of keras numpy image tensor.
  """
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  def download_file(url):
    req = urllib2.Request(url)
    response = urllib2.urlopen(req)
    return response
  response = download_file(url)
  tmpfile = BytesIO()
  while True:
    # Download a piece of the file from the connection
    s = response.read(16384)
    # Once the entire file has been downloaded, tarfile returns b''
    # (the empty bytes) which is a falsey value
    if not s:
      break
    # Otherwise, write the piece of the file to the temporary file.
    tmpfile.write(s)
  response.close()

  tmpfile.seek(0)
  tar_dir = tarfile.open(mode='r:gz', fileobj=tmpfile)
  X = None
  y = None
  for member in tar_dir.getnames():
    if '_batch' in member:
      filestream = tar_dir.extractfile(member).read()
      batch = pickle.load(StringIO.StringIO(filestream))
      if X is None:
        X = np.array(batch['data'], dtype=np.uint8)
        y = np.array(batch['labels'])
      else:
        X = np.concatenate((X, np.array(batch['data'], dtype=np.uint8)))
        y = np.concatenate((y, np.array(batch['labels'])))
  data = Dataset(X, y)
  return data


def get_mldata(dataset):
  # Use scikit to grab datasets and save them save_dir.
  save_dir = FLAGS.save_dir
  filename = os.path.join(save_dir, dataset[1]+'.pkl')

  if not gfile.Exists(save_dir):
    gfile.MkDir(save_dir)
  if not gfile.Exists(filename):
    if dataset[0][-3:] == 'csv':
      data = get_csv_data(dataset[0])
    elif dataset[0] == 'breast_cancer':
      data = load_breast_cancer()
    elif dataset[0] == 'iris':
      data = load_iris()
    elif dataset[0] == 'newsgroup':
      # Removing header information to make sure that no newsgroup identifying
      # information is included in data
      data = fetch_20newsgroups_vectorized(subset='all', remove=('headers'))
      tfidf = TfidfTransformer(norm='l2')
      X = tfidf.fit_transform(data.data)
      data.data = X
    elif dataset[0] == 'rcv1':
      sklearn.datasets.rcv1.URL = (
        'http://www.ai.mit.edu/projects/jmlr/papers/'
        'volume5/lewis04a/a13-vector-files/lyrl2004_vectors')
      sklearn.datasets.rcv1.URL_topics = (
        'http://www.ai.mit.edu/projects/jmlr/papers/'
        'volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz')
      data = sklearn.datasets.fetch_rcv1(
          data_home='/tmp')
    elif dataset[0] == 'wikipedia_attack':
      data = get_wikipedia_talk_data()
    elif dataset[0] == 'cifar10':
      data = get_cifar10()
    elif 'keras' in dataset[0]:
      data = get_keras_data(dataset[0])
    else:
      try:
        data = fetch_mldata(dataset[0])
      except:
        raise Exception('ERROR: failed to fetch data from mldata.org')
    X = data.data
    y = data.target
    if X.shape[0] != y.shape[0]:
      X = np.transpose(X)
    assert X.shape[0] == y.shape[0]

    data = {'data': X, 'target': y}
    pickle.dump(data, gfile.GFile(filename, 'w'))


def main(argv):
  del argv  # Unused.
  # First entry of tuple is mldata.org name, second is the name that we'll use
  # to reference the data.
  datasets = [('mnist (original)', 'mnist'), ('australian', 'australian'),
              ('heart', 'heart'), ('breast_cancer', 'breast_cancer'),
              ('iris', 'iris'), ('vehicle', 'vehicle'), ('wine', 'wine'),
              ('waveform ida', 'waveform'), ('german ida', 'german'),
              ('splice ida', 'splice'), ('ringnorm ida', 'ringnorm'),
              ('twonorm ida', 'twonorm'), ('diabetes_scale', 'diabetes'),
              ('mushrooms', 'mushrooms'), ('letter', 'letter'), ('dna', 'dna'),
              ('banana-ida', 'banana'), ('letter', 'letter'), ('dna', 'dna'),
              ('newsgroup', 'newsgroup'), ('cifar10', 'cifar10'),
              ('cifar10_keras', 'cifar10_keras'),
              ('cifar100_keras', 'cifar100_keras'),
              ('cifar100_coarse_keras', 'cifar100_coarse_keras'),
              ('mnist_keras', 'mnist_keras'),
              ('wikipedia_attack', 'wikipedia_attack'),
              ('rcv1', 'rcv1')]

  if FLAGS.datasets:
    subset = FLAGS.datasets.split(',')
    datasets = [d for d in datasets if d[1] in subset]

  for d in datasets:
    print(d[1])
    get_mldata(d)


if __name__ == '__main__':
  app.run()
