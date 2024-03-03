import numpy as np

from structured_segmentation.layers import StructuredClassifierLayer
from structured_segmentation.layers import InputLayer
from structured_segmentation.layers import VotingLayer
from structured_segmentation.layers import NormalizationLayer
from structured_segmentation.layers import BottleNeckLayer

from structured_segmentation.model.model import Model
from structured_segmentation.model.graph import Graph


def randomize_options(max_val):
    k_y = max(np.random.randint(max_val), 1)
    k_x = int(max_val - k_y)
    return k_y, k_x


class RandomEnsemble(Model):
    def __init__(self,
                 image_width=None,
                 image_height=None,
                 initial_image_down_scale=None,
                 output_aggregation_options="boosting",
                 n_estimators=5,
                 max_depth=1,
                 max_kernel_sum=10,
                 max_stride_sum=1,
                 max_down_scale=6,
                 features_to_use="gray-color",
                 norm_input=None,
                 clf="mlp_x",
                 kernel_shape="ellipse",
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

        self.clf = clf
        self.clf_options = clf_options
        super(RandomEnsemble, self).__init__()

        self.model = self.build(
            image_width,
            image_height,
            initial_image_down_scale,
            output_aggregation_options
        )

    def build(self, width=None, height=None, initial_down_scale=None, output_option="voting"):
        trees = []
        for i in range(self.n_estimators):
            tree = InputLayer(
                name="input_tree_{}".format(i),
                features_to_use=self.features_to_use,
                width=width, height=height,
                initial_down_scale=initial_down_scale
            )

            if self.norm_input is not None:
                tree = NormalizationLayer(
                    INPUTS=tree,
                    name="norm_tree_{}".format(i),
                    norm_option=self.norm_input
                )

            tree = StructuredClassifierLayer(
                INPUTS=tree,
                name="tree_{}".format(i),
                kernel=randomize_options(self.max_kernel_sum),
                strides=randomize_options(self.max_stride_sum),
                kernel_shape=self.kernel_shape,
                down_scale=np.random.randint(self.max_down_scale),
                data_reduction=self.data_reduction,
                clf=self.clf,
                clf_options=self.clf_options
            )
            trees.append(tree)

        if output_option == "voting":
            trees = VotingLayer(INPUTS=trees, name="voting")
        elif output_option == "boosting":
            b = BottleNeckLayer(INPUTS=trees, name="cls_preparation")
            trees = StructuredClassifierLayer(INPUTS=b, name="boosting", clf=self.clf, clf_options=self.clf_options)
        return Graph(layer_stack=trees)
