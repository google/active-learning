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

"""Another informative and diverse sampler that mirrors the algorithm described
in Xu, et. al., Representative Sampling for Text Classification Using 
Support Vector Machines, 2003

Batch is created by clustering points within the margin of the classifier and 
choosing points closest to the k centroids.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sampling_methods.sampling_def import SamplingMethod


class RepresentativeClusterMeanSampling(SamplingMethod):
  """Selects batch based on informative and diverse criteria.

    Returns points within the margin of the classifier that are closest to the
    k-means centers of those points.  
  """

  def __init__(self, X, y, seed):
    self.name = 'cluster_mean'
    self.X = X
    self.flat_X = self.flatten_X()
    self.y = y
    self.seed = seed

  def select_batch_(self, model, N, already_selected, **kwargs):
    # Probably okay to always use MiniBatchKMeans
    # Should standardize data before clustering
    # Can cluster on standardized data but train on raw features if desired
    try:
      distances = model.decision_function(self.X)
    except:
      distances = model.predict_proba(self.X)
    if len(distances.shape) < 2:
      min_margin = abs(distances)
    else:
      sort_distances = np.sort(distances, 1)[:, -2:]
      min_margin = sort_distances[:, 1] - sort_distances[:, 0]
    rank_ind = np.argsort(min_margin)
    rank_ind = [i for i in rank_ind if i not in already_selected]

    distances = abs(model.decision_function(self.X))
    min_margin_by_class = np.min(abs(distances[already_selected]),axis=0)
    unlabeled_in_margin = np.array([i for i in range(len(self.y))
                                    if i not in already_selected and
                                    any(distances[i]<min_margin_by_class)])
    if len(unlabeled_in_margin) < N:
      print("Not enough points within margin of classifier, using simple uncertainty sampling")
      return rank_ind[0:N]
    clustering_model = MiniBatchKMeans(n_clusters=N)
    dist_to_centroid = clustering_model.fit_transform(self.flat_X[unlabeled_in_margin])
    medoids = np.argmin(dist_to_centroid,axis=0)
    medoids = list(set(medoids))
    selected_indices = unlabeled_in_margin[medoids]
    selected_indices = sorted(selected_indices,key=lambda x: min_margin[x])
    remaining = [i for i in rank_ind if i not in selected_indices]
    selected_indices.extend(remaining[0:N-len(selected_indices)])
    return selected_indices
