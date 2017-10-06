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

"""Hierarchical cluster AL method.

Implements algorithm described in Dasgupta, S and Hsu, D,
"Hierarchical Sampling for Active Learning, 2008
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sampling_methods.sampling_def import SamplingMethod
from sampling_methods.utils.tree import Tree


class HierarchicalClusterAL(SamplingMethod):
  """Implements hierarchical cluster AL based method.

  All methods are internal.  select_batch_ is called via abstract classes
  outward facing method select_batch.

  Default affininity is euclidean and default linkage is ward which links
  cluster based on variance reduction.  Hence, good results depend on
  having normalized and standardized data.
  """

  def __init__(self, X, y, seed, beta=2, affinity='euclidean', linkage='ward',
               clustering=None, max_features=None):
    """Initializes AL method and fits hierarchical cluster to data.

    Args:
      X: data
      y: labels for determinining number of clusters as an input to
        AgglomerativeClustering
      seed: random seed used for sampling datapoints for batch
      beta: width of error used to decide admissble labels, higher value of beta
        corresponds to wider confidence and less stringent definition of
        admissibility
        See scikit Aggloerative clustering method for more info
      affinity: distance metric used for hierarchical clustering
      linkage: linkage method used to determine when to join clusters
      clustering: can provide an AgglomerativeClustering that is already fit
      max_features: limit number of features used to construct hierarchical
        cluster.  If specified, PCA is used to perform feature reduction and
        the hierarchical clustering is performed using transformed features.
    """
    self.name = 'hierarchical'
    self.seed = seed
    np.random.seed(seed)
    # Variables for the hierarchical cluster
    self.already_clustered = False
    if clustering is not None:
      self.model = clustering
      self.already_clustered = True
    self.n_leaves = None
    self.n_components = None
    self.children_list = None
    self.node_dict = None
    self.root = None  # Node name, all node instances access through self.tree
    self.tree = None
    # Variables for the AL algorithm
    self.initialized = False
    self.beta = beta
    self.labels = {}
    self.pruning = []
    self.admissible = {}
    self.selected_nodes = None
    # Data variables
    self.classes = None
    self.X = X

    classes = list(set(y))
    self.n_classes = len(classes)
    if max_features is not None:
      transformer = PCA(n_components=max_features)
      transformer.fit(X)
      self.transformed_X = transformer.fit_transform(X)
      #connectivity = kneighbors_graph(self.transformed_X,max_features)
      self.model = AgglomerativeClustering(
          affinity=affinity, linkage=linkage, n_clusters=len(classes))
      self.fit_cluster(self.transformed_X)
    else:
      self.model = AgglomerativeClustering(
          affinity=affinity, linkage=linkage, n_clusters=len(classes))
      self.fit_cluster(self.X)
    self.y = y

    self.y_labels = {}
    # Fit cluster and update cluster variables

    self.create_tree()
    print('Finished creating hierarchical cluster')

  def fit_cluster(self, X):
    if not self.already_clustered:
      self.model.fit(X)
      self.already_clustered = True
    self.n_leaves = self.model.n_leaves_
    self.n_components = self.model.n_components_
    self.children_list = self.model.children_

  def create_tree(self):
    node_dict = {}
    for i in range(self.n_leaves):
      node_dict[i] = [None, None]
    for i in range(len(self.children_list)):
      node_dict[self.n_leaves + i] = self.children_list[i]
    self.node_dict = node_dict
    # The sklearn hierarchical clustering algo numbers leaves which correspond
    # to actual datapoints 0 to n_points - 1 and all internal nodes have
    # ids greater than n_points - 1 with the root having the highest node id
    self.root = max(self.node_dict.keys())
    self.tree = Tree(self.root, self.node_dict)
    self.tree.create_child_leaves_mapping(range(self.n_leaves))
    for v in node_dict:
      self.admissible[v] = set()

  def get_child_leaves(self, node):
    return self.tree.get_child_leaves(node)

  def get_node_leaf_counts(self, node_list):
    node_counts = []
    for v in node_list:
      node_counts.append(len(self.get_child_leaves(v)))
    return np.array(node_counts)

  def get_class_counts(self, y):
    """Gets the count of all classes in a sample.

    Args:
      y: sample vector for which to perform the count
    Returns:
      count of classes for the sample vector y, the class order for count will
      be the same as that of self.classes
    """
    unique, counts = np.unique(y, return_counts=True)
    complete_counts = []
    for c in self.classes:
      if c not in unique:
        complete_counts.append(0)
      else:
        index = np.where(unique == c)[0][0]
        complete_counts.append(counts[index])
    return np.array(complete_counts)

  def observe_labels(self, labeled):
    for i in labeled:
      self.y_labels[i] = labeled[i]
    self.classes = np.array(
        sorted(list(set([self.y_labels[k] for k in self.y_labels]))))
    self.n_classes = len(self.classes)

  def initialize_algo(self):
    self.pruning = [self.root]
    self.labels[self.root] = np.random.choice(self.classes)
    node = self.tree.get_node(self.root)
    node.best_label = self.labels[self.root]
    self.selected_nodes = [self.root]

  def get_node_class_probabilities(self, node, y=None):
    children = self.get_child_leaves(node)
    if y is None:
      y_dict = self.y_labels
    else:
      y_dict = dict(zip(range(len(y)), y))
    labels = [y_dict[c] for c in children if c in y_dict]
    # If no labels have been observed, simply return uniform distribution
    if len(labels) == 0:
      return 0, np.ones(self.n_classes)/self.n_classes
    return len(labels), self.get_class_counts(labels) / (len(labels) * 1.0)

  def get_node_upper_lower_bounds(self, node):
    n_v, p_v = self.get_node_class_probabilities(node)
    # If no observations, return worst possible upper lower bounds
    if n_v == 0:
      return np.zeros(len(p_v)), np.ones(len(p_v))
    delta = 1. / n_v + np.sqrt(p_v * (1 - p_v) / (1. * n_v))
    return (np.maximum(p_v - delta, np.zeros(len(p_v))),
            np.minimum(p_v + delta, np.ones(len(p_v))))

  def get_node_admissibility(self, node):
    p_lb, p_up = self.get_node_upper_lower_bounds(node)
    all_other_min = np.vectorize(
        lambda i:min([1 - p_up[c] for c in range(len(self.classes)) if c != i]))
    lowest_alternative_error = self.beta * all_other_min(
        np.arange(len(self.classes)))
    return 1 - p_lb < lowest_alternative_error

  def get_adjusted_error(self, node):
    _, prob = self.get_node_class_probabilities(node)
    error = 1 - prob
    admissible = self.get_node_admissibility(node)
    not_admissible = np.where(admissible != True)[0]
    error[not_admissible] = 1.0
    return error

  def get_class_probability_pruning(self, method='lower'):
    prob_pruning = []
    for v in self.pruning:
      label = self.labels[v]
      label_ind = np.where(self.classes == label)[0][0]
      if method == 'empirical':
        _, v_prob = self.get_node_class_probabilities(v)
      else:
        lower, upper = self.get_node_upper_lower_bounds(v)
        if method == 'lower':
          v_prob = lower
        elif method == 'upper':
          v_prob = upper
        else:
          raise NotImplementedError
      prob = v_prob[label_ind]
      prob_pruning.append(prob)
    return np.array(prob_pruning)

  def get_pruning_impurity(self, y):
    impurity = []
    for v in self.pruning:
      _, prob = self.get_node_class_probabilities(v, y)
      impurity.append(1-max(prob))
    impurity = np.array(impurity)
    weights = self.get_node_leaf_counts(self.pruning)
    weights = weights / sum(weights)
    return sum(impurity*weights)

  def update_scores(self):
    node_list = set(range(self.n_leaves))
    # Loop through generations from bottom to top
    while len(node_list) > 0:
      parents = set()
      for v in node_list:
        node = self.tree.get_node(v)
        # Update admissible labels for node
        admissible = self.get_node_admissibility(v)
        admissable_indices = np.where(admissible)[0]
        for l in self.classes[admissable_indices]:
          self.admissible[v].add(l)
        # Calculate score
        v_error = self.get_adjusted_error(v)
        best_label_ind = np.argmin(v_error)
        if admissible[best_label_ind]:
          node.best_label = self.classes[best_label_ind]
        score = v_error[best_label_ind]
        node.split = False

        # Determine if node should be split
        if v >= self.n_leaves:  # v is not a leaf
          if len(admissable_indices) > 0:  # There exists an admissible label
            # Make sure label set for node so that we can flow to children
            # if necessary
            assert node.best_label is not None
            # Only split if all ancestors are admissible nodes
            # This is part  of definition of admissible pruning
            admissible_ancestors = [len(self.admissible[a]) > 0 for a in
                                    self.tree.get_ancestor(node)]
            if all(admissible_ancestors):
              left = self.node_dict[v][0]
              left_node = self.tree.get_node(left)
              right = self.node_dict[v][1]
              right_node = self.tree.get_node(right)
              node_counts = self.get_node_leaf_counts([v, left, right])
              split_score = (node_counts[1] / node_counts[0] *
                             left_node.score + node_counts[2] /
                             node_counts[0] * right_node.score)
              if split_score < score:
                score = split_score
                node.split = True
        node.score = score
        if node.parent:
          parents.add(node.parent.name)
        node_list = parents

  def update_pruning_labels(self):
    for v in self.selected_nodes:
      node = self.tree.get_node(v)
      pruning = self.tree.get_pruning(node)
      self.pruning.remove(v)
      self.pruning.extend(pruning)
    # Check that pruning covers all leave nodes
    node_counts = self.get_node_leaf_counts(self.pruning)
    assert sum(node_counts) == self.n_leaves
    # Fill in labels
    for v in self.pruning:
      node = self.tree.get_node(v)
      if node.best_label  is None:
        node.best_label = node.parent.best_label
      self.labels[v] = node.best_label

  def get_fake_labels(self):
    fake_y = np.zeros(self.X.shape[0])
    for p in self.pruning:
      indices = self.get_child_leaves(p)
      fake_y[indices] = self.labels[p]
    return fake_y

  def train_using_fake_labels(self, model, X_test, y_test):
    classes_labeled = set([self.labels[p] for p in self.pruning])
    if len(classes_labeled) == self.n_classes:
      fake_y = self.get_fake_labels()
      model.fit(self.X, fake_y)
      test_acc = model.score(X_test, y_test)
      return test_acc
    return 0

  def select_batch_(self, N, already_selected, labeled, y, **kwargs):
    # Observe labels for previously recommended batches
    self.observe_labels(labeled)

    if not self.initialized:
      self.initialize_algo()
      self.initialized = True
      print('Initialized algo')

    print('Updating scores and pruning for labels from last batch')
    self.update_scores()
    self.update_pruning_labels()
    print('Nodes in pruning: %d' % (len(self.pruning)))
    print('Actual impurity for pruning is: %.2f' %
          (self.get_pruning_impurity(y)))

    # TODO(lishal): implement multiple selection methods
    selected_nodes = set()
    weights = self.get_node_leaf_counts(self.pruning)
    probs = 1 - self.get_class_probability_pruning()
    weights = weights * probs
    weights = weights / sum(weights)
    batch = []

    print('Sampling batch')
    while len(batch) < N:
      node = np.random.choice(list(self.pruning), p=weights)
      children = self.get_child_leaves(node)
      children = [
          c for c in children if c not in self.y_labels and c not in batch
      ]
      if len(children) > 0:
        selected_nodes.add(node)
        batch.append(np.random.choice(children))
    self.selected_nodes = selected_nodes
    return batch

  def to_dict(self):
    output = {}
    output['node_dict'] = self.node_dict
    return output
