import os
import joblib
from time import time

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from structured_segmentation.utils.utils import check_n_make_dir, save_dict, load_dict


def init_mlp(clf):
    if clf == "mlp":
        return MLPRegressor(hidden_layer_sizes=(64, ), max_iter=100000)
    elif clf == "mlp_x":
        return MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=100000)
    elif clf == "mlp_xx":
        return MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=100000)
    raise Exception("Unknown specifications for classifier: {}".format(clf))


def init_rf(clf):
    if clf == "rf":
        return RandomForestRegressor(n_estimators=100, n_jobs=-1)
    elif clf == "rf_10":
        return RandomForestRegressor(n_estimators=10, n_jobs=-1)
    elif clf == "rf_200":
        return RandomForestRegressor(n_estimators=200, n_jobs=-1)
    elif clf == "rf_500":
        return RandomForestRegressor(n_estimators=500, n_jobs=-1)
    raise Exception("Unknown specifications for classifier: {}".format(clf))


def regressor_initialize(opt):
    if "rf" in opt["type"]:
        return init_rf(opt["type"])
    elif "mlp" in opt["type"]:
        return init_mlp(opt["type"])
    elif opt["type"] == "extra_tree":
        return ExtraTreesRegressor(n_estimators=100, n_jobs=-1)
    else:
        raise ValueError("type: {} not recognised".format(opt["type"]))


class RegressorHandler:
    def __init__(self, opt=None):
        self.opt = opt

        self.regressor = None

        self.best_params = None
        self.best_score = None

    def __str__(self):
        return "Regressor: {}".format(self.opt["type"])

    def fit(self, x_train, y_train):
        print("Fitting the {} to the training set".format(self.opt["type"]))
        t0 = time()
        self.regressor.fit(x_train, y_train)
        print("done in %0.3fs" % (time() - t0))

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

    def new_regressor(self):
        self.regressor = regressor_initialize(self.opt)

    def load(self, model_path, name="clf"):
        self.opt = load_dict(os.path.join(model_path, "{}_opt.json".format(name)))
        self.regressor = joblib.load(os.path.join(model_path, "{}.pkl".format(name)))

    def save(self, model_path, name="clf"):
        check_n_make_dir(model_path)
        save_dict(self.opt, os.path.join(model_path, "{}_opt.json".format(name)))
        if self.regressor is not None:
            joblib.dump(self.regressor, os.path.join(model_path, "{}.pkl".format(name)))
