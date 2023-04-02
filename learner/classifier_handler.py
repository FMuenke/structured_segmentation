import os
import joblib
import numpy as np
from time import time

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import MiniBatchKMeans

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from utils.utils import check_n_make_dir, save_dict, load_dict


def set_default_options(opt):
    default_options = {
        "n_estimators": 100,
        "max_iter": 100000,
        "layer_structure": (100, ),
        "n_clusters": 8
    }
    for def_opt in default_options:
        if def_opt not in opt:
            opt[def_opt] = default_options[def_opt]
    return opt


def classifier_initialize(opt):
    opt = set_default_options(opt)
    if opt["type"] in ["random_forrest", "rf"]:
        return RandomForestClassifier(n_estimators=opt["n_estimators"], class_weight="balanced", n_jobs=-1)
    elif opt["type"] == "ada_boost":
        return AdaBoostClassifier(n_estimators=opt["n_estimators"])
    elif opt["type"] in ["logistic_regression", "lr"]:
        return LogisticRegression(class_weight='balanced', max_iter=opt["max_iter"])
    elif opt["type"] == "sgd":
        return SGDClassifier(class_weight='balanced', max_iter=opt["max_iter"])
    elif opt["type"] in ["support_vector_machine", "svm"]:
        return SVC(kernel='rbf', class_weight='balanced', gamma="scale")
    elif opt["type"] in ["multilayer_perceptron", "mlp"]:
        return MLPClassifier(hidden_layer_sizes=opt["layer_structure"], max_iter=opt["max_iter"])
    elif opt["type"] in ["mlp_x"]:
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=opt["max_iter"])
    elif opt["type"] in ["decision_tree", "dt", "tree"]:
        return DecisionTreeClassifier()
    elif opt["type"] in ["b_decision_tree", "b_dt", "b_tree"]:
        return DecisionTreeClassifier(class_weight="balanced")
    elif opt["type"] in ["neighbours", "knn"]:
        return KNeighborsClassifier(n_neighbors=opt["n_neighbours"])
    elif opt["type"] == "extra_tree":
        return ExtraTreesClassifier(n_estimators=opt["n_estimators"], class_weight="balanced", n_jobs=-1)
    elif opt["type"] == "kmeans":
        return MiniBatchKMeans(n_clusters=opt["n_clusters"])
    elif opt["type"] == "lr_cv":
        return LogisticRegressionCV(max_iter=opt["max_iter"], class_weight="balanced")
    else:
        raise ValueError("type: {} not recognised".format(opt["type"]))


class ClassifierHandler:
    def __init__(self, opt=None):
        self.opt = opt

        self.classifier = None

        self.best_params = None
        self.best_score = None

        self.report = None

    def __str__(self):
        return "Classifier: {}".format(self.opt["type"])

    def fit(self, x_train, y_train):
        print("[INFO] Fitting the {} to the training set (N Features: {})".format(
            self.opt["type"], x_train.shape[1]))
        t0 = time()
        self.classifier.fit(x_train, y_train)
        print("[INFO] done in %0.3fs" % (time() - t0))

    def fit_inc_hyper_parameter(self, x, y, param_set, scoring='f1_macro', cv=3, n_iter=None, n_jobs=-1):
        if self.classifier is None:
            self.new_classifier()
        if n_iter is None:
            print("[INFO] Starting GridSearchCV:")
            searcher = GridSearchCV(
                self.classifier,
                param_set, scoring=scoring, n_jobs=n_jobs, cv=cv, verbose=1, refit=True)
        else:
            print("[INFO] Starting RandomizedSearchCV:")
            searcher = RandomizedSearchCV(self.classifier, param_set, n_iter=n_iter, cv=cv, verbose=1,
                                          random_state=42, n_jobs=n_jobs, scoring=scoring, refit=True)
        searcher.fit(x, y)
        self.best_params = searcher.best_params_
        self.best_score = searcher.best_score_
        self.classifier = searcher.best_estimator_

    def predict(self, x):
        if self.opt["type"] == "kmeans":
            x = x.astype(np.float32)
        return self.classifier.predict(x)

    def predict_proba(self, x):
        if hasattr(self.classifier, "predict_proba"):
            prob_pos = self.classifier.predict_proba(x)
        else:  # use decision function
            prob_pos = self.classifier.predict(x)
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
                self.report += str(f1_score(y_true=y_test, y_pred=y_pred, average="micro", zero_division=0))
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

    def new_classifier(self):
        self.classifier = classifier_initialize(self.opt)

    def load(self, model_path, name="clf"):
        self.opt = load_dict(os.path.join(model_path, "{}_opt.json".format(name)))
        self.classifier = joblib.load(os.path.join(model_path, "{}.pkl".format(name)))

    def save(self, model_path, name="clf"):
        check_n_make_dir(model_path)
        save_dict(self.opt, os.path.join(model_path, "{}_opt.json".format(name)))
        if self.classifier is not None:
            joblib.dump(self.classifier, os.path.join(model_path, "{}.pkl".format(name)))
        if self.report is not None:
            with open(os.path.join(model_path, "classification_report.txt"), "w") as f:
                f.write(self.report)

    def is_fitted(self):
        try:
            if self.classifier is not None:
                return check_is_fitted(self.classifier)
            else:
                return False
        except Exception as e:
            return False

