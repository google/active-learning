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

"""Abstract class for wrapper sampling methods that call base sampling methods.

Provides interface to sampling methods that allow same signature
for select_batch.  Each subclass implements select_batch_ with the desired
signature for readability.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from sampling_methods.constants import AL_MAPPING
from sampling_methods.constants import get_all_possible_arms
from sampling_methods.sampling_def import SamplingMethod

get_all_possible_arms()


class WrapperSamplingMethod(SamplingMethod):
  __metaclass__ = abc.ABCMeta

  def initialize_samplers(self, mixtures):
    methods = []
    for m in mixtures:
      methods += m['methods']
    methods = set(methods)
    self.base_samplers = {}
    for s in methods:
      self.base_samplers[s] = AL_MAPPING[s](self.X, self.y, self.seed)
    self.samplers = []
    for m in mixtures:
      self.samplers.append(
          AL_MAPPING['mixture_of_samplers'](self.X, self.y, self.seed, m,
                                            self.base_samplers))
