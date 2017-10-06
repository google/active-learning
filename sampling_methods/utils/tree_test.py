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

"""Tests for sampling_methods.utils.tree."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from sampling_methods.utils import tree


class TreeTest(unittest.TestCase):

  def setUp(self):
    node_dict = {
        1: (2, 3),
        2: (4, 5),
        3: (6, 7),
        4: [None, None],
        5: [None, None],
        6: [None, None],
        7: [None, None]
    }
    self.tree = tree.Tree(1, node_dict)
    self.tree.create_child_leaves_mapping([4, 5, 6, 7])
    node = self.tree.get_node(1)
    node.split = True
    node = self.tree.get_node(2)
    node.split = True

  def assertNode(self, node, name, left, right):
    self.assertEqual(node.name, name)
    self.assertEqual(node.left.name, left)
    self.assertEqual(node.right.name, right)

  def testTreeRootSetCorrectly(self):
    self.assertNode(self.tree.root, 1, 2, 3)

  def testGetNode(self):
    node = self.tree.get_node(1)
    assert isinstance(node, tree.Node)
    self.assertEqual(node.name, 1)

  def testFillParent(self):
    node = self.tree.get_node(3)
    self.assertEqual(node.parent.name, 1)

  def testGetAncestors(self):
    ancestors = self.tree.get_ancestor(5)
    self.assertTrue(all([a in ancestors for a in [1, 2]]))

  def testChildLeaves(self):
    leaves = self.tree.get_child_leaves(3)
    self.assertTrue(all([c in leaves for c in [6, 7]]))

  def testFillWeights(self):
    node = self.tree.get_node(3)
    self.assertEqual(node.weight, 0.5)

  def testGetPruning(self):
    node = self.tree.get_node(1)
    pruning = self.tree.get_pruning(node)
    self.assertTrue(all([n in pruning for n in [3, 4, 5]]))

if __name__ == '__main__':
  unittest.main()
