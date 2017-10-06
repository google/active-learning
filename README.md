# Active Learning Playground

## Introduction

This is a python module for experimenting with different active learning
algorithms. There are a few key components to running active learning
experiments:

*   Main experiment script is
    [`run_experiment.py`](run_experiment.py)
    with many flags for different run options.

*   Supported datasets can be downloaded to a specified directory by running
    [`utils/create_data.py`](utils/create_data.py).

*   Supported active learning methods are in
    [`sampling_methods`](sampling_methods/).

Below I will go into each component in more detail.

DISCLAIMER: This is not an official Google product.

## Setup
The dependencies are in [`requirements.txt`](requirements.txt).  Please make sure these packages are
installed before running experiments.  If GPU capable `tensorflow` is desired, please follow
instructions [here](https://www.tensorflow.org/install/).

It is highly suggested that you install all dependencies into a separate `virtualenv` for
easy package management.

## Getting benchmark datasets

By default the datasets are saved to `/tmp/data`. You can specify another directory via the
`--save_dir` flag.

Redownloading all the datasets will be very time consuming so please be patient.
You can specify a subset of the data to download by passing in a comma separated
string of datasets via the `--datasets` flag.

## Running experiments

There are a few key flags for
[`run_experiment.py`](run_experiment.py):

*   `dataset`: name of the dataset, must match the save name used in
    `create_data.py`. Must also exist in the data_dir.

*   `sampling_method`: active learning method to use. Must be specified in
    [`sampling_methods/constants.py`](sampling_methods/constants.py).

*   `warmstart_size`: initial batch of uniformly sampled examples to use as seed
    data. Float indicates percentage of total training data and integer
    indicates raw size.

*   `batch_size`: number of datapoints to request in each batch. Float indicates
    percentage of total training data and integer indicates raw size.

*   `score_method`: model to use to evaluate the performance of the sampling
    method. Must be in `get_model` method of
    [`utils/utils.py`](utils/utils.py).

*   `data_dir`: directory with saved datasets.

*   `save_dir`: directory to save results.

This is just a subset of all the flags. There are also options for
preprocessing, introducing labeling noise, dataset subsampling, and using a
different model to select than to score/evaluate.

## Available active learning methods

All named active learning methods are in
[`sampling_methods/constants.py`](sampling_methods/constants.py).

You can also specify a mixture of active learning methods by following the
pattern of `[sampling_method]-[mixture_weight]` separated by dashes; i.e.
`mixture_of_samplers-margin-0.33-informative_diverse-0.33-uniform-0.34`.

Some supported sampling methods include:

*   Uniform: samples are selected via uniform sampling.

*   Margin: uncertainty based sampling method.

*   Informative and diverse: margin and cluster based sampling method.

*   k-center greedy: representative strategy that greedily forms a batch of
    points to minimize maximum distance from a labeled point.

*   Graph density: representative strategy that selects points in dense regions
    of pool.

*   Exp3 bandit: meta-active learning method that tries to learns optimal
    sampling method using a popular multi-armed bandit algorithm.

### Adding new active learning methods

Implement either a base sampler that inherits from
[`SamplingMethod`](sampling_methods/sampling_def.py)
or a meta-sampler that calls base samplers which inherits from
[`WrapperSamplingMethod`](sampling_methods/wrapper_sampler_def.py).

The only method that must be implemented by any sampler is `select_batch_`,
which can have arbitrary named arguments. The only restriction is that the name
for the same input must be consistent across all the samplers (i.e. the indices
for already selected examples all have the same name across samplers). Adding a
new named argument that hasn't been used in other sampling methods will require
feeding that into the `select_batch` call in
[`run_experiment.py`](run_experiment.py).

After implementing your sampler, be sure to add it to
[`constants.py`](sampling_methods/constants.py)
so that it can be called from
[`run_experiment.py`](run_experiment.py).

## Available models

All available models are in the `get_model` method of
[`utils/utils.py`](utils/utils.py).

Supported methods:

*   Linear SVM: scikit method with grid search wrapper for regularization
    parameter.

*   Kernel SVM: scikit method with grid search wrapper for regularization
    parameter.

*   Logistc Regression: scikit method with grid search wrapper for
    regularization parameter.

*   Small CNN: 4 layer CNN optimized using rmsprop implemented in Keras with
    tensorflow backend.

*   Kernel Least Squares Classification: block gradient descient solver that can
    use multiple cores so is often faster than scikit Kernel SVM.

### Adding new models

New models must follow the scikit learn api and implement the following methods

*   `fit(X, y[, sample_weight])`: fit the model to the input features and
    target.

*   `predict(X)`: predict the value of the input features.

*   `score(X, y)`: returns target metric given test features and test targets.

*   `decision_function(X)` (optional): return class probabilities, distance to
    decision boundaries, or other metric that can be used by margin sampler as a
    measure of uncertainty.

See
[`small_cnn.py`](utils/small_cnn.py)
for an example.

After implementing your new model, be sure to add it to `get_model` method of
[`utils/utils.py`](utils/utils.py).

Currently models must be added on a one-off basis and not all scikit-learn
classifiers are supported due to the need for user input on whether and how to
tune the hyperparameters of the model. However, it is very easy to add a
scikit-learn model with hyperparameter search wrapped around as a supported
model.

## Collecting results and charting

The
[`utils/chart_data.py`](utils/chart_data.py)
script handles processing of data and charting for a specified dataset and
source directory.
