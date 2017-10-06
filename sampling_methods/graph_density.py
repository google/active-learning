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

"""Diversity promoting sampling method that uses graph density to determine
 most representative points.

This is an implementation of the method described in
https://www.mpi-inf.mpg.de/fileadmin/inf/d2/Research_projects_files/EbertCVPR2012.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
import numpy as np
from sampling_methods.sampling_def import SamplingMethod


class GraphDensitySampler(SamplingMethod):
  """Diversity promoting sampling method that uses graph density to determine
  most representative points.
  """

  def __init__(self, X, y, seed):
    self.name = 'graph_density'
    self.X = X
    self.flat_X = self.flatten_X()
    # Set gamma for gaussian kernel to be equal to 1/n_features
    self.gamma = 1. / self.X.shape[1]
    self.compute_graph_density()

  def compute_graph_density(self, n_neighbor=10):
    # kneighbors graph is constructed using k=10
    connect = kneighbors_graph(self.flat_X, n_neighbor,p=1)
    # Make connectivity matrix symmetric, if a point is a k nearest neighbor of
    # another point, make it vice versa
    neighbors = connect.nonzero()
    inds = zip(neighbors[0],neighbors[1])
    # Graph edges are weighted by applying gaussian kernel to manhattan dist.
    # By default, gamma for rbf kernel is equal to 1/n_features but may
    # get better results if gamma is tuned.
    for entry in inds:
      i = entry[0]
      j = entry[1]
      distance = pairwise_distances(self.flat_X[[i]],self.flat_X[[j]],metric='manhattan')
      distance = distance[0,0]
      weight = np.exp(-distance * self.gamma)
      connect[i,j] = weight
      connect[j,i] = weight
    self.connect = connect
    # Define graph density for an observation to be sum of weights for all
    # edges to the node representing the datapoint.  Normalize sum weights
    # by total number of neighbors.
    self.graph_density = np.zeros(self.X.shape[0])
    for i in np.arange(self.X.shape[0]):
      self.graph_density[i] = connect[i,:].sum() / (connect[i,:]>0).sum()
    self.starting_density = copy.deepcopy(self.graph_density)

  def select_batch_(self, N, already_selected, **kwargs):
    # If a neighbor has already been sampled, reduce the graph density
    # for its direct neighbors to promote diversity.
    batch = set()
    self.graph_density[already_selected] = min(self.graph_density) - 1
    while len(batch) < N:
      selected = np.argmax(self.graph_density)
      neighbors = (self.connect[selected,:] > 0).nonzero()[1]
      self.graph_density[neighbors] = self.graph_density[neighbors] - self.graph_density[selected]
      batch.add(selected)
      self.graph_density[already_selected] = min(self.graph_density) - 1
      self.graph_density[list(batch)] = min(self.graph_density) - 1
    return list(batch)

  def to_dict(self):
    output = {}
    output['connectivity'] = self.connect
    output['graph_density'] = self.starting_density
    return output