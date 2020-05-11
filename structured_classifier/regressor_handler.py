import os
import joblib
from time import time

from xgboost import XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.utils import check_n_make_dir, save_dict, load_dict


class RegressorHandler:
    def __init__(self, opt=None):
        self.opt = opt

        self.regressor = None

        self.best_params = None
        self.best_score = None

    def __str__(self):
        s = ""
        s += "Regressor: {}".format(self.opt["type"])
        return s

    def fit(self, x_train, y_train):
        print("Fitting the {} to the training set".format(self.opt["type"]))
        t0 = time()
        self.regressor.fit(x_train, y_train)
        print("done in %0.3fs" % (time() - t0))

    def fit_inc_hyper_parameter(self, x, y, param_set, cv=3, n_iter=None, n_jobs=-1):
        if self.regressor is None:
            self.new_regressor()
        if n_iter is None:
            print(" ")
            print("Starting GridSearchCV:")
            searcher = GridSearchCV(self.regressor, param_set, n_jobs=n_jobs, cv=cv, verbose=10, refit=True)
        else:
            print(" ")
            print("Starting RandomizedSearchCV:")
            searcher = RandomizedSearchCV(self.regressor, param_set, n_iter=n_iter, cv=cv, verbose=10,
                                          random_state=42, n_jobs=n_jobs, refit=True)
        searcher.fit(x, y)
        self.best_params = searcher.best_params_
        self.best_score = searcher.best_score_
        self.regressor = searcher.best_estimator_

    def predict(self, x):
        return self.regressor.predict(x)

    def predict_proba(self, x):
        return self.regressor.predict_proba(x)

    def evaluate(self, x_test, y_test, save_path=None):
        print("Predicting on the test set")
        t0 = time()
        y_pred = self.predict(x_test)
        print("done in %0.3fs" % (time() - t0))

        print("Regression Results:")
        print("MAE: " + str(mean_squared_error(y_test, y_pred)))
        print("MSE: " + str(mean_absolute_error(y_test, y_pred)))

    def _init_regressor(self, opt):
        if "base_estimator" in opt:
            b_est = self._init_regressor({"classifier_opt": opt["base_estimator"]})
        else:
            b_est = None

        if "n_estimators" in opt:
            n_estimators = opt["n_estimators"]
        else:
            n_estimators = 200

        if "num_parallel_tree" in opt:
            num_parallel_tree = opt["num_parallel_tree"]
        else:
            num_parallel_tree = 5

        if "max_iter" in opt:
            max_iter = opt["max_iter"]
        else:
            max_iter = 1000

        if opt["type"] in ["random_forrest", "rf"]:
            return RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1)
        if opt["type"] in ["sgd"]:
            return MultiOutputRegressor(SGDRegressor(max_iter=max_iter))
        if opt["type"] in ["gradient_boosting", "gb"]:
            return MultiOutputRegressor(GradientBoostingRegressor(n_estimators=n_estimators))
        if opt["type"] in ["tree", "decision_tree", "dt"]:
            return DecisionTreeRegressor()
        if opt["type"] in ["mlp"]:
            return MLPRegressor(max_iter=max_iter)
        elif opt["type"] == "ada_boost":
            return MultiOutputRegressor(AdaBoostRegressor(base_estimator=b_est, n_estimators=n_estimators))
        elif opt["type"] in ["extra_tree", "extra"]:
            return ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1)
        elif opt["type"] in ["xgboost", "xgb"]:
            return MultiOutputRegressor(XGBRegressor(
                n_estimators=n_estimators,
                objective="reg:squarederror",
                num_parallel_tree=num_parallel_tree,
                tree_method="hist",
                booster="gbtree",
                n_jobs=-1))
        else:
            raise ValueError("type: {} not recognised".format(opt["type"]))

    def new_regressor(self):
        self.regressor = self._init_regressor(self.opt)

    def load(self, model_path, name="clf"):
        self.opt = load_dict(os.path.join(model_path, "{}_opt.json".format(name)))
        self.regressor = joblib.load(os.path.join(model_path, "{}.pkl".format(name)))

    def save(self, model_path, name="clf"):
        check_n_make_dir(model_path)
        save_dict(self.opt, os.path.join(model_path, "{}_opt.json".format(name)))
        if self.regressor is not None:
            joblib.dump(self.regressor, os.path.join(model_path, "{}.pkl".format(name)))

