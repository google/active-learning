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

"""Mixture of base sampling strategies

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from sampling_methods.sampling_def import SamplingMethod
from sampling_methods.constants import AL_MAPPING, get_base_AL_mapping

get_base_AL_mapping()


class MixtureOfSamplers(SamplingMethod):
  """Samples according to mixture of base sampling methods.

  If duplicate points are selected by the mixed strategies when forming the batch
  then the remaining slots are divided according to mixture weights and
  another partial batch is requested until the batch is full.
  """
  def __init__(self,
               X,
               y,
               seed,
               mixture={'methods': ('margin', 'uniform'),
                        'weight': (0.5, 0.5)},
               samplers=None):
    self.X = X
    self.y = y
    self.name = 'mixture_of_samplers'
    self.sampling_methods = mixture['methods']
    self.sampling_weights = dict(zip(mixture['methods'], mixture['weights']))
    self.seed = seed
    # A list of initialized samplers is allowed as an input because
    # for AL_methods that search over different mixtures, may want mixtures to
    # have shared AL_methods so that initialization is only performed once for
    # computation intensive methods like HierarchicalClusteringAL and
    # states are shared between mixtures.
    # If initialized samplers are not provided, initialize them ourselves.
    if samplers is None:
      self.samplers = {}
      self.initialize(self.sampling_methods)
    else:
      self.samplers = samplers
    self.history = []

  def initialize(self, samplers):
    self.samplers = {}
    for s in samplers:
      self.samplers[s] = AL_MAPPING[s](self.X, self.y, self.seed)

  def select_batch_(self, already_selected, N, **kwargs):
    """Returns batch of datapoints selected according to mixture weights.

    Args:
      already_included: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to add using margin active learner
    """
    kwargs['already_selected'] = copy.copy(already_selected)
    inds = set()
    self.selected_by_sampler = {}
    for s in self.sampling_methods:
      self.selected_by_sampler[s] = []
    effective_N = 0
    while len(inds) < N:
      effective_N += N - len(inds)
      for s in self.sampling_methods:
        if len(inds) < N:
          batch_size = min(max(int(self.sampling_weights[s] * effective_N), 1), N)
          sampler = self.samplers[s]
          kwargs['N'] = batch_size
          s_inds = sampler.select_batch(**kwargs)
          for ind in s_inds:
            if ind not in self.selected_by_sampler[s]:
              self.selected_by_sampler[s].append(ind)
          s_inds = [d for d in s_inds if d not in inds]
          s_inds = s_inds[0 : min(len(s_inds), N-len(inds))]
          inds.update(s_inds)
    self.history.append(copy.deepcopy(self.selected_by_sampler))
    return list(inds)

  def to_dict(self):
    output = {}
    output['history'] = self.history
    output['samplers'] = self.sampling_methods
    output['mixture_weights'] = self.sampling_weights
    for s in self.samplers:
      s_output = self.samplers[s].to_dict()
      output[s] = s_output
    return output
