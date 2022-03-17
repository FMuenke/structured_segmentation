import numpy as np


from structured_classifier.experimental.graph_3d_layer import Graph3DLayer
from structured_classifier.experimental.input_3d_layer import Input3DLayer
from structured_classifier.experimental.voting_3d_layer import Voting3DLayer
from structured_classifier.experimental.bottle_neck_3d_layer import BottleNeck3DLayer

from structured_classifier.pixel_layer import PixelLayer
from structured_classifier.input_layer import InputLayer
from structured_classifier.voting_layer import VotingLayer
from structured_classifier.normalization_layer import NormalizationLayer
from structured_classifier.bottle_neck_layer import BottleNeckLayer


from model.base_structures import get_decision_layer, get_decision_layer_3d


class RandomStructuredRandomForrest3D:
    def __init__(self,
                 n_estimators,
                 max_depth=1,
                 max_kernel_sum=25,
                 max_down_scale=6,
                 features_to_use="gray-color",
                 tree_type="kernel",
                 feature_aggregation="hist32",
                 clf="b_rf",
                 clf_options=None,
                 kernel_shape="square",
                 data_reduction=3):
        self.features_to_use = features_to_use
        self.n_estimators = n_estimators
        self.max_kernel_sum = max_kernel_sum
        self.max_depth = max_depth
        self.max_down_scale = max_down_scale
        self.data_reduction = data_reduction
        self.kernel_shape = kernel_shape
        self.tree_type = tree_type
        self.feature_aggregation = feature_aggregation

        self.clf = clf
        self.clf_options = clf_options

    def build(self, width=None, height=None, initial_down_scale=None, output_option="voting"):
        trees = []
        for i in range(self.n_estimators):
            tree = Input3DLayer(name="input_tree_{}".format(i),
                                features_to_use=self.features_to_use,
                                width=width, height=height,
                                initial_down_scale=initial_down_scale)

            for ii in range(self.max_depth):
                k_t = max(np.random.randint(self.max_kernel_sum), 1)
                k_x = max(np.random.randint(self.max_kernel_sum - k_t), 1)
                k_y = max(int(self.max_kernel_sum - k_t - k_x), 1)
                k = (k_t, k_y, k_x)
                if type(self.clf) is list:
                    clf = np.random.choice(self.clf)
                else:
                    clf = self.clf
                d = np.random.randint(self.max_down_scale)
                tree = get_decision_layer_3d(
                    INPUTS=tree,
                    name="tree_{}_{}".format(i, ii),
                    decision_type=self.tree_type,
                    kernel=k, kernel_shape=self.kernel_shape,
                    feature_aggregation=self.feature_aggregation,
                    down_scale=d, data_reduction=self.data_reduction,
                    clf=clf, clf_options=self.clf_options
                )
            trees.append(tree)

        if output_option == "voting":
            return Voting3DLayer(INPUTS=trees, name="voting")
        elif output_option == "boosting":
            b = BottleNeck3DLayer(INPUTS=trees, name="cls_preparation")
            return Graph3DLayer(INPUTS=b, name="boosting", clf=self.clf, clf_options=self.clf_options)
        else:
            return trees


class RandomStructuredRandomForrest:
    def __init__(self,
                 n_estimators=10,
                 max_depth=1,
                 max_kernel_sum=10,
                 max_stride_sum=5,
                 max_down_scale=6,
                 features_to_use="gray-color",
                 tree_type="kernel",
                 feature_aggregation="hist32",
                 norm_input=None,
                 clf="b_rf",
                 kernel_shape="square",
                 clf_options=None,
                 data_reduction=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_kernel_sum = max_kernel_sum
        self.max_stride_sum = max_stride_sum
        self.max_down_scale = max_down_scale
        self.features_to_use = features_to_use
        self.norm_input = norm_input
        self.kernel_shape = kernel_shape
        self.data_reduction = data_reduction
        self.tree_type = tree_type
        self.feature_aggregation = feature_aggregation

        self.clf = clf
        self.clf_options = clf_options

    def build(self, width=None, height=None, initial_down_scale=None, output_option="voting"):
        trees = []
        for i in range(self.n_estimators):
            if type(self.features_to_use) is list:
                features_to_use = np.random.choice(self.features_to_use)
            else:
                features_to_use = self.features_to_use
            tree = InputLayer(name="input_tree_{}".format(i),
                              features_to_use=features_to_use,
                              width=width, height=height,
                              initial_down_scale=initial_down_scale)

            if self.norm_input is not None:
                tree = NormalizationLayer(INPUTS=tree,
                                          name="norm_tree_{}".format(i),
                                          norm_option=self.norm_input)

            for ii in range(self.max_depth):
                k_y = max(np.random.randint(self.max_kernel_sum), 1)
                k_x = int(self.max_kernel_sum - k_y)
                k = (k_y, k_x)

                s_y = max(np.random.randint(self.max_kernel_sum), 1)
                s_x = max(np.random.randint(self.max_kernel_sum), 1)
                s = (s_y, s_x)

                if type(self.clf) is list:
                    clf = np.random.choice(self.clf)
                else:
                    clf = self.clf

                d = np.random.randint(self.max_down_scale)

                tree = get_decision_layer(
                    INPUTS=tree,
                    name="tree_{}_{}".format(i, ii),
                    decision_type=self.tree_type,
                    kernel=k, strides=s, kernel_shape=self.kernel_shape,
                    feature_aggregation=self.feature_aggregation,
                    down_scale=d, data_reduction=self.data_reduction,
                    clf=clf, clf_options=self.clf_options
                )
            trees.append(tree)

        if output_option == "voting":
            return VotingLayer(INPUTS=trees, name="voting")
        elif output_option == "boosting":
            b = BottleNeckLayer(INPUTS=trees, name="cls_preparation")
            return PixelLayer(INPUTS=b, name="boosting", clf=self.clf, clf_options=self.clf_options)
        return trees
