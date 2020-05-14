import numpy as np


from structured_classifier.decision_3d_layer import Decision3DLayer
from structured_classifier.input_3d_layer import Input3DLayer
from structured_classifier.voting_3d_layer import Voting3DLayer

from structured_classifier.decision_layer import DecisionLayer
from structured_classifier.input_layer import InputLayer
from structured_classifier.voting_layer import VotingLayer
from structured_classifier.normalization_layer import NormalizationLayer


class RandomStructuredRandomForrest3D:
    def __init__(self, n_estimators, max_kernel_sum, max_down_scale, clf="b_rf", clf_options=None, data_reduction=3):
        self.n_estimators = n_estimators
        self.max_kernel_sum = max_kernel_sum
        self.max_down_scale = max_down_scale
        self.data_reduction = data_reduction

        self.clf = clf
        self.clf_options = clf_options

    def build(self, width=None, height=None, initial_down_scale=None, include_voting=True):
        trees = []
        for i in range(self.n_estimators):
            tree_in = Input3DLayer(name="input_tree_{}".format(i),
                                   features_to_use=["gray-color"],
                                   width=width, height=height,
                                   initial_down_scale=initial_down_scale)
            k_t = np.random.randint(self.max_kernel_sum)
            k_y = np.random.randint(self.max_kernel_sum - k_t)
            k_x = np.random.randint(self.max_kernel_sum - k_t - k_y)
            k = (k_t, k_y, k_x)
            print(k)
            if type(self.clf) is list:
                clf = np.random.choice(self.clf)
            else:
                clf = self.clf
            d = np.random.randint(self.max_down_scale)
            tree = Decision3DLayer(INPUTS=tree_in,
                                   name="tree_{}".format(i),
                                   kernel=k, kernel_shape="ellipse",
                                   down_scale=d, data_reduction=self.data_reduction,
                                   clf=clf, clf_options=self.clf_options)

            trees.append(tree)

        if include_voting:
            return Voting3DLayer(INPUTS=trees, name="voting")
        return trees


class RandomStructuredRandomForrest:
    def __init__(self,
                 n_estimators,
                 max_kernel_sum,
                 max_down_scale,
                 norm_input=None,
                 clf="b_rf",
                 clf_options=None,
                 data_reduction=3):
        self.n_estimators = n_estimators
        self.max_kernel_sum = max_kernel_sum
        self.max_down_scale = max_down_scale
        self.norm_input = norm_input
        self.data_reduction = data_reduction

        self.clf = clf
        self.clf_options = clf_options

    def build(self, width=None, height=None, initial_down_scale=None, include_voting=True):
        trees = []
        for i in range(self.n_estimators):
            tree_in = InputLayer(name="input_tree_{}".format(i),
                                 features_to_use=["gray-color"],
                                 width=width, height=height,
                                 initial_down_scale=initial_down_scale)
            if self.norm_input is not None:
                tree_in = NormalizationLayer(INPUTS=tree_in,
                                             name="norm_tree_{}".format(i),
                                             norm_option=self.norm_input)
            k_y = np.random.randint(self.max_kernel_sum) + 1
            k_x = np.random.randint(self.max_kernel_sum - k_y + 1) + 1
            k = (k_y, k_x)
            if type(self.clf) is list:
                clf = np.random.choice(self.clf)
            else:
                clf = self.clf
            d = np.random.randint(self.max_down_scale)
            tree = DecisionLayer(INPUTS=tree_in,
                                 name="tree_{}".format(i),
                                 kernel=k, kernel_shape="ellipse",
                                 down_scale=d, data_reduction=self.data_reduction,
                                 clf=clf, clf_options=self.clf_options)

            trees.append(tree)

        if include_voting:
            return VotingLayer(INPUTS=trees, name="voting")
        return trees
