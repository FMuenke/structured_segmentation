import os
import joblib
from time import time
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding

from sklearn.utils.validation import check_is_fitted

from structured_segmentation.utils.utils import check_n_make_dir, save_dict, load_dict
from structured_segmentation.learner.internal_classifier import InternalClassifier


class Origami:
    def __init__(self) -> None:
        pass

    def fit(self, x, y):
        return
    
    def transform(self, x):
        return x


def encoder_initialize(opt):
    if opt["type"] == "pca_4":
        return PCA(n_components=4)
    elif opt["type"] == "pca_8":
        return PCA(n_components=8)
    elif opt["type"] == "pca_16":
        return PCA(n_components=16)
    elif opt["type"] == "iso_4":
        return Isomap(n_components=4, n_jobs=-1)
    elif opt["type"] == "iso_8":
        return Isomap(n_components=8, n_jobs=-1)
    elif opt["type"] == "iso_16":
        return Isomap(n_components=16, n_jobs=-1)
    elif opt["type"] == "locally_linear_4":
        return LocallyLinearEmbedding(n_components=4, n_jobs=-1)
    elif opt["type"] == "locally_linear_8":
        return LocallyLinearEmbedding(n_components=8, n_jobs=-1)
    elif opt["type"] == "locally_linear_16":
        return LocallyLinearEmbedding(n_components=16, n_jobs=-1)
    elif opt["type"] == "origami":
        return Origami()
    else:
        raise ValueError("type: {} not recognised".format(opt["type"]))


class InternalEncoder(InternalClassifier):
    def __init__(self, opt=None):
        super(InternalEncoder, self).__init__(opt)

        def __str__(self):
            return "ENC: {}".format(self.opt["type"])

    def predict_proba(self, x):
        if hasattr(self.model, "predict_proba"):
            prob_pos = self.model.predict_proba(x)
        elif hasattr(self.model, "transform"):
            x = x.astype(np.float32)
            prob_pos = self.model.transform(x)
        else:  # use decision function
            prob_pos = self.model.predict(x)
            prob_pos = np.expand_dims(prob_pos, axis=1)
        return prob_pos
    
    def transform(self, x):
        return self.model.transform(x)

    def new(self):
        self.model = encoder_initialize(self.opt)

    def evaluate(self, x_test, y_test, save_path=None, verbose=True):
        pass