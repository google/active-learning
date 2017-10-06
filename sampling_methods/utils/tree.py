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

"""Node and Tree class to support hierarchical clustering AL method.

Assumed to be binary tree.

Node class is used to represent each node in a hierarchical clustering.
Each node has certain properties that are used in the AL method.

Tree class is used to traverse a hierarchical clustering.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy


class Node(object):
  """Node class for hierarchical clustering.

  Initialized with name and left right children.
  """

  def __init__(self, name, left=None, right=None):
    self.name = name
    self.left = left
    self.right = right
    self.is_leaf = left is None and right is None
    self.parent = None
    # Fields for hierarchical clustering AL
    self.score = 1.0
    self.split = False
    self.best_label = None
    self.weight = None

  def set_parent(self, parent):
    self.parent = parent


class Tree(object):
  """Tree object for traversing a binary tree.

  Most methods apply to trees in general with the exception of get_pruning
  which is specific to the hierarchical clustering AL method.
  """

  def __init__(self, root, node_dict):
    """Initializes tree and creates all nodes in node_dict.

    Args:
      root: id of the root node
      node_dict: dictionary with node_id as keys and entries indicating
        left and right child of node respectively.
    """
    self.node_dict = node_dict
    self.root = self.make_tree(root)
    self.nodes = {}
    self.leaves_mapping = {}
    self.fill_parents()
    self.n_leaves = None

  def print_tree(self, node, max_depth):
    """Helper function to print out tree for debugging."""
    node_list = [node]
    output = ""
    level = 0
    while level < max_depth and len(node_list):
      children = set()
      for n in node_list:
        node = self.get_node(n)
        output += ("\t"*level+"node %d: score %.2f, weight %.2f" %
                   (node.name, node.score, node.weight)+"\n")
        if node.left:
          children.add(node.left.name)
        if node.right:
          children.add(node.right.name)
      level += 1
      node_list = children
    return print(output)

  def make_tree(self, node_id):
    if node_id is not None:
      return Node(node_id,
                  self.make_tree(self.node_dict[node_id][0]),
                  self.make_tree(self.node_dict[node_id][1]))

  def fill_parents(self):
    # Setting parent and storing nodes in dict for fast access
    def rec(pointer, parent):
      if pointer is not None:
        self.nodes[pointer.name] = pointer
        pointer.set_parent(parent)
        rec(pointer.left, pointer)
        rec(pointer.right, pointer)
    rec(self.root, None)

  def get_node(self, node_id):
    return self.nodes[node_id]

  def get_ancestor(self, node):
    ancestors = []
    if isinstance(node, int):
      node = self.get_node(node)
    while node.name != self.root.name:
      node = node.parent
      ancestors.append(node.name)
    return ancestors

  def fill_weights(self):
    for v in self.node_dict:
      node = self.get_node(v)
      node.weight = len(self.leaves_mapping[v]) / (1.0 * self.n_leaves)

  def create_child_leaves_mapping(self, leaves):
    """DP for creating child leaves mapping.
    
    Storing in dict to save recompute.
    """
    self.n_leaves = len(leaves)
    for v in leaves:
      self.leaves_mapping[v] = [v]
    node_list = set([self.get_node(v).parent for v in leaves])
    while node_list:
      to_fill = copy.copy(node_list)
      for v in node_list:
        if (v.left.name in self.leaves_mapping
            and v.right.name in self.leaves_mapping):
          to_fill.remove(v)
          self.leaves_mapping[v.name] = (self.leaves_mapping[v.left.name] +
                                         self.leaves_mapping[v.right.name])
          if v.parent is not None:
            to_fill.add(v.parent)
      node_list = to_fill
    self.fill_weights()

  def get_child_leaves(self, node):
    return self.leaves_mapping[node]

  def get_pruning(self, node):
    if node.split:
      return self.get_pruning(node.left) + self.get_pruning(node.right)
    else:
      return [node.name]

