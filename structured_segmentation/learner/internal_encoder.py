import os
import joblib
from time import time

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding

from sklearn.utils.validation import check_is_fitted

from structured_segmentation.utils.utils import check_n_make_dir, save_dict, load_dict


class EncoderHandler:
    def __init__(self, opt=None):
        self.opt = opt

        self.encoder = None

        self.best_params = None
        self.best_score = None

    def __str__(self):
        s = ""
        s += "Encoder: {}".format(self.opt["type"])
        return s

    def fit(self, x_train, y_train):
        print("Fitting the {} to the training set".format(self.opt["type"]))
        t0 = time()
        self.encoder.fit(x_train, None)
        print("done in %0.3fs" % (time() - t0))

    def transform(self, x):
        return self.encoder.transform(x)

    def _init_encoder(self, opt):
        if "n_components" in opt:
            n_components = opt["n_components"]
        else:
            n_components = 8

        if opt["type"] == "pca":
            return PCA(n_components=n_components)
        elif opt["type"] == "kernel_pca":
            return KernelPCA(n_components=n_components, n_jobs=-1)
        elif opt["type"] == "iso":
            return Isomap(n_components=n_components, n_jobs=-1)
        elif opt["type"] == "locally_linear":
            return LocallyLinearEmbedding(n_components=n_components, n_jobs=-1)
        else:
            raise ValueError("type: {} not recognised".format(opt["type"]))

    def new_encoder(self):
        self.encoder = self._init_encoder(self.opt)

    def load(self, model_path, name="clf"):
        self.opt = load_dict(os.path.join(model_path, "{}_opt.json".format(name)))
        self.encoder = joblib.load(os.path.join(model_path, "{}.pkl".format(name)))

    def save(self, model_path, name="clf"):
        check_n_make_dir(model_path)
        save_dict(self.opt, os.path.join(model_path, "{}_opt.json".format(name)))
        if self.encoder is not None:
            joblib.dump(self.encoder, os.path.join(model_path, "{}.pkl".format(name)))

    def is_fitted(self):
        if self.encoder is not None:
            return check_is_fitted(self.encoder)
        else:
            return False
