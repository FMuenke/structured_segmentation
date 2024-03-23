import os
import joblib
import numpy as np
from time import time
import logging

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from structured_segmentation.utils.utils import check_n_make_dir, save_dict, load_dict


def init_mlp(clf):
    if clf == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, ), max_iter=100000)
    elif clf == "mlp_x":
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100000)
    elif clf == "mlp_xx":
        return MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=100000)
    raise Exception("Unknown specifications for classifier: {}".format(clf))


def init_rf(clf):
    if clf == "rf":
        return RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1)
    elif clf == "rf_10":
        return RandomForestClassifier(n_estimators=10, class_weight="balanced", n_jobs=-1)
    elif clf == "rf_200":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1)
    elif clf == "rf_500":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1)
    raise Exception("Unknown specifications for classifier: {}".format(clf))


def init_lr(clf):
    if clf == "lr":
        return LogisticRegression(class_weight='balanced', max_iter=100000)
    elif clf == "lr_cv":
        return LogisticRegressionCV(max_iter=100000, class_weight="balanced")
    raise Exception("Unknown specifications for classifier: {}".format(clf))


def init_kmeans(clf):
    name, n_clusters = clf.split("_")
    return MiniBatchKMeans(n_clusters=int(n_clusters), n_init="auto")


def classifier_initialize(opt):
    if "rf" in opt["type"]:
        return init_rf(opt["type"])
    elif "lr" in opt["type"]:
        return init_lr(opt["type"])
    elif "mlp" in opt["type"]:
        return init_mlp(opt["type"])
    elif "lda" in opt["type"]:
        return LinearDiscriminantAnalysis()
    elif opt["type"] == "nb":
        return GaussianNB()
    elif opt["type"] == "sgd":
        return SGDClassifier(
            class_weight="balanced", loss="log_loss", penalty="l1", max_iter=10000, n_jobs=-1)
    elif opt["type"] == "hgb":
        return HistGradientBoostingClassifier(max_iter=1000, class_weight="balanced")
    elif opt["type"] == "extra_tree":
        return ExtraTreesClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1)
    elif "kmeans" in opt["type"]:
        return init_kmeans(opt["type"])
    else:
        raise ValueError("type: {} not recognised".format(opt["type"]))


class InternalClassifier:
    def __init__(self, opt=None):
        self.opt = opt
        self.model = None
        self.best_params = None
        self.best_score = None
        self.report = None

    def __str__(self):
        return "CLF: {}".format(self.opt["type"])

    def fit(self, x_train, y_train):
        print("[INFO] Fitting - {} (Samples: {} / Features: {})".format(
            self.opt["type"], x_train.shape[0], x_train.shape[1]))
        t0 = time()

        # Remove classes with too few samples
        n_samples = 4
        unique_labels, label_counts = np.unique(y_train, return_counts=True)
        labels_to_remove = unique_labels[label_counts < n_samples]
        indices_to_remove = np.isin(y_train, labels_to_remove)
        
        x_train = x_train[~indices_to_remove]
        y_train = y_train[~indices_to_remove]
        self.model.fit(x_train, y_train)
        print("[INFO] done in %0.3fs" % (time() - t0))

    def predict(self, x):
        if self.opt["type"] == "kmeans":
            x = x.astype(np.float32)
        t0 = time()
        result = self.model.predict(x)
        logging.info("[INFO] Prediction done in %0.3fs" % (time() - t0))
        return result

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

    def evaluate(self, x_test, y_test, save_path=None, verbose=True):
        if verbose:
            print("[INFO] Predicting on the test set")
        t0 = time()
        y_pred = self.predict(x_test)
        self.report = str(classification_report(y_test, y_pred, zero_division=0))
        if len(y_pred.shape) > 1:
            if y_pred.shape[1] > 10:
                self.report = "Macro F1-Score:\n"
                self.report += str(f1_score(
                    y_true=y_test, y_pred=y_pred, average="micro", zero_division=0))
        if verbose:
            print("[INFO] done in %0.3fs" % (time() - t0))
            print(self.report)
        if save_path is not None:
            d = os.path.dirname(save_path)
            if not os.path.isdir(d):
                os.mkdir(d)
            with open(save_path, "w") as f:
                f.write(self.report)

        return f1_score(y_true=y_test, y_pred=y_pred, average="macro", zero_division=0)

    def new(self):
        self.model = classifier_initialize(self.opt)

    def load(self, model_path, name="clf"):
        self.opt = load_dict(os.path.join(model_path, "{}_opt.json".format(name)))
        self.model = joblib.load(os.path.join(model_path, "{}.pkl".format(name)))

    def save(self, model_path, name="clf"):
        check_n_make_dir(model_path)
        save_dict(self.opt, os.path.join(model_path, "{}_opt.json".format(name)))
        if self.model is not None:
            joblib.dump(self.model, os.path.join(model_path, "{}.pkl".format(name)))
        if self.report is not None:
            with open(os.path.join(model_path, "classification_report.txt"), "w") as f:
                f.write(self.report)
