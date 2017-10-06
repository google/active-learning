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

"""Uniform sampling method.

Samples in batches.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sampling_methods.sampling_def import SamplingMethod


class UniformSampling(SamplingMethod):

  def __init__(self, X, y, seed):
    self.X = X
    self.y = y
    self.name = 'uniform'
    np.random.seed(seed)

  def select_batch_(self, already_selected, N, **kwargs):
    """Returns batch of randomly sampled datapoints.

    Assumes that data has already been shuffled.

    Args:
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to label
    """

    # This is uniform given the remaining pool but biased wrt the entire pool.
    sample = [i for i in range(self.X.shape[0]) if i not in already_selected]
    return sample[0:N]
