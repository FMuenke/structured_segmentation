import os
import joblib
import numpy as np
from time import time

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import RUSBoostClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from utils.utils import check_n_make_dir, save_dict, load_dict
from utils.plot import plot_calibration_curve


class ClassifierHandler:
    def __init__(self, model_path, opt=None):
        self.model_path = model_path
        self.opt = opt

        self.classifier = None

        self.best_params = None
        self.best_score = None

        self.path_to_pipeline_opt = os.path.join(self.model_path, "pipeline_opt.json")
        self.path_to_classifier = os.path.join(self.model_path, "classifier.pkl")

    def __str__(self):
        s = ""
        s += "Classifier: {}, Score: {}\n".format(self.opt["classifier_opt"]["type"], self.best_score)
        s += "Parameters: \n"
        s += "{}\n".format(self.best_params)
        return s

    def fit(self, x_train, y_train):
        print("Fitting the {} to the training set".format(self.opt["classifier_opt"]["type"]))
        t0 = time()
        self.classifier.fit(x_train, y_train)
        print("done in %0.3fs" % (time() - t0))

    def fit_inc_hyper_parameter(self, x, y, param_set, scoring='f1_macro', cv=3, n_iter=None, n_jobs=-1):
        if self.classifier is None:
            self.new_classifier()
        if n_iter is None:
            print(" ")
            print("Starting GridSearchCV:")
            searcher = GridSearchCV(self.classifier, param_set, scoring, n_jobs=n_jobs, cv=cv, verbose=10, refit=True)
        else:
            print(" ")
            print("Starting RandomizedSearchCV:")
            searcher = RandomizedSearchCV(self.classifier, param_set, n_iter=n_iter, cv=cv, verbose=10,
                                          random_state=42, n_jobs=n_jobs, scoring=scoring, refit=True)
        searcher.fit(x, y)
        self.best_params = searcher.best_params_
        self.best_score = searcher.best_score_
        self.classifier = searcher.best_estimator_

    def calibrate(self, x_train, y_train, x_test, y_test, scoring):
        print("Calibrating Classifier...")
        t0 = time()
        est = self.classifier
        est.fit(x_train, y_train)
        isotonic = CalibratedClassifierCV(est, cv=5, method='isotonic')
        isotonic.fit(x_train, y_train)
        sigmoid = CalibratedClassifierCV(est, cv=5, method='sigmoid')
        sigmoid.fit(x_train, y_train)

        est_list = [(est, "base"), (isotonic, "isotonic"), (sigmoid, "sigmoid")]
        score = 0
        for clf, name in est_list:
            clf_score = scoring(y_test, clf.predict(x_test))
            if clf_score > score:
                score = clf_score
                self.classifier = clf
        print("done in %0.3fs" % (time() - t0))

        plot_calibration_curve(est_list, x_test, y_test, path=self.model_path)

    def predict(self, x, get_confidence=False):
        if get_confidence:
            try:
                prob = self.classifier.predict_proba(x)
                return [np.argmax(prob)], np.max(prob)
            except:
                return self.classifier.predict(x), 1.00
        return self.classifier.predict(x)

    def evaluate(self, x_test, y_test, save_path=None):
        print("Predicting on the test set")
        t0 = time()
        y_pred = self.predict(x_test)
        print("done in %0.3fs" % (time() - t0))

        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        if save_path is not None:
            d = os.path.dirname(save_path)
            if not os.path.isdir(d):
                os.mkdir(d)
            with open(save_path, "w") as f:
                f.write(classification_report(y_test, y_pred))

        return f1_score(y_true=y_test, y_pred=y_pred, average="macro")

    def _init_classifier(self, pipeline_opt):
        opt = pipeline_opt["classifier_opt"]
        if opt["type"] == "random_forrest":
            if "n_estimators" in opt:
                return RandomForestClassifier(n_estimators=opt["n_estimators"], class_weight="balanced", n_jobs=-1)
            else:
                return RandomForestClassifier(class_weight="balanced", n_jobs=-1)
        elif opt["type"] == "ada_boost":
            if "base_estimator" in opt:
                b_est = self._init_classifier({"classifier_opt": opt["base_estimator"]})
            else:
                b_est = None
            if "n_estimators" in opt:
                return AdaBoostClassifier(base_estimator=b_est, n_estimators=opt["n_estimators"])
            else:
                return AdaBoostClassifier(base_estimator=b_est)
        elif opt["type"] in ["logistic_regression", "lr"]:
            return LogisticRegression()
        elif opt["type"] == "sgd":
            return SGDClassifier()
        elif opt["type"] in ["bernoulli_bayes", "bayes"]:
            return BernoulliNB()
        elif opt["type"] in ["support_vector_machine", "svm"]:
            return SVC(kernel='rbf', class_weight='balanced', gamma="scale")
        elif opt["type"] in ["multilayer_perceptron", "mlp"]:
            return MLPClassifier()
        elif opt["type"] == "decision_tree":
            return DecisionTreeClassifier()
        elif opt["type"] in ["neighbours", "knn"]:
            return KNeighborsClassifier(n_neighbors=opt["n_neighbours"])
        elif opt["type"] == "extra_tree":
            if "n_estimators" in opt:
                return ExtraTreesClassifier(n_estimators=opt["n_estimators"], class_weight="balanced", n_jobs=-1)
            else:
                return ExtraTreesClassifier(class_weight="balanced", n_jobs=-1)
        elif opt["type"] == "xgboost":
            return XGBClassifier(objective='binary:logistic', n_jobs=-1)
        elif opt["type"] in ["b_random_forrest", "rf"]:
            return BalancedRandomForestClassifier(n_jobs=-1)
        elif opt["type"] == "b_bagging":
            if "base_estimator" in opt:
                b_est = self._init_classifier({"classifier_opt": opt["base_estimator"]})
            else:
                b_est = None
            return BalancedBaggingClassifier(base_estimator=b_est)
        elif opt["type"] == "b_boosting":
            if "base_estimator" in opt:
                b_est = self._init_classifier({"classifier_opt": opt["base_estimator"]})
            else:
                b_est = None
            return RUSBoostClassifier(base_estimator=b_est)
        else:
            raise ValueError("type: {} not recognised".format(opt["type"]))

    def new_classifier(self):
        if "classifier_opt" in self.opt:
            self.classifier = self._init_classifier(self.opt)
        else:
            raise ValueError("No Classifier Option was defined.")

    def load(self):
        self.opt = load_dict(self.path_to_pipeline_opt)
        self.classifier = joblib.load(os.path.join(self.model_path, "classifier.pkl"))
        print("Classifier was loaded!")

    def save(self):
        check_n_make_dir(self.model_path, clean=True)
        save_dict(self.opt, self.path_to_pipeline_opt)
        if self.classifier is not None:
            joblib.dump(self.classifier, os.path.join(self.model_path, "classifier.pkl"))

        print("machine_learning-Pipeline was saved to: {}".format(self.model_path))

