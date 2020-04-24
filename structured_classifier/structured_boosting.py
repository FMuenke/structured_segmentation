import os
import numpy as np
from tqdm import tqdm

from structured_classifier.classifier_handler import ClassifierHandler
from features.feature_extractor import FeatureExtractor

from utils.utils import save_dict, load_dict


class StructuredBoosting:
    def __init__(self, model_folder):
        self.model_folder = model_folder

        self.opt_path = os.path.join(self.model_folder, "config.json")
        self.clf_path = os.path.join(self.model_folder, "clf")

        self.opt = None
        self.structured_classifier = None
        self.fex = None
        self.clf = None

    def build(self, cfg, structured_classifier):
        self.structured_classifier = structured_classifier
        self.opt = cfg.opt
        self.opt["features_to_use"] = ["raw"]
        self.opt["look_up_window_gradient"] = None
        self.fex = FeatureExtractor(self.opt)
        self.clf = ClassifierHandler(self.clf_path, self.opt)
        self.clf.new_classifier()

    def save(self):
        save_dict(self.opt, self.opt_path)
        self.clf.save()

    def load(self, structured_classifier):
        self.structured_classifier = structured_classifier
        self.opt = load_dict(self.opt_path)
        self.fex = FeatureExtractor(self.opt)
        self.clf = ClassifierHandler(self.clf_path)
        self.clf.load()

    def extract_features(self, image):
        x_img = self.structured_classifier.inference(image, interpolation="linear")
        return self.fex.extract(x_img)

    def get_x_y(self, tag_set):
        x = []
        y = []
        print("Feature Extraction...")
        for t in tqdm(tag_set):
            x_img = t.load_x()
            x_img = self.extract_features(x_img)
            h_img, w_img = x_img.shape[:2]
            y_img = t.load_y([h_img, w_img])

            x_img = np.reshape(x_img, (h_img * w_img, -1))
            y_img = np.reshape(y_img, h_img * w_img)

            x.append(x_img)
            y.append(y_img)
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        return x, y

    def fit(self, train_set, validation_set, data_reduction_factor=None):
        x_train, y_train = self.get_x_y(train_set)
        x_val, y_val = self.get_x_y(validation_set)

        if data_reduction_factor is not None:
            x_train, y_train = x_train[::data_reduction_factor, :], y_train[::data_reduction_factor]

        n_samples_train, n_features = x_train.shape
        n_samples_val = x_val.shape[0]
        print("DataSet has {} Samples (Train: {}/ Validation: {}) with {} features.".format(
            n_samples_train + n_samples_val, n_samples_train, n_samples_val, n_features
        ))
        if "param_grid" in self.opt["classifier_opt"]:
            param_set = self.opt["classifier_opt"]["param_grid"]
            self.clf.fit_inc_hyper_parameter(x_train, y_train, param_set, n_iter=300)
        else:
            self.clf.fit(x_train, y_train)
        self.clf.evaluate(x_val, y_val, save_path=os.path.join(self.model_folder, "classifier_report.txt"))