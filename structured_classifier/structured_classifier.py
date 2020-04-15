import os
import cv2
import numpy as np
from tqdm import tqdm
from data_structure.image_handler import ImageHandler
from utils.utils import save_dict, load_dict
from features.feature_extractor import FeatureExtractor

from structured_classifier.classifier_handler import ClassifierHandler


class StructuredClassifier:
    def __init__(self, model_folder):
        self.model_folder = model_folder
        self.opt_path = os.path.join(self.model_folder, "config.json")
        self.clf_path = os.path.join(self.model_folder, "clf")

        self.opt = None
        self.fex = None
        self.clf = None

    def build(self, cfg):
        self.opt = cfg.opt
        self.fex = FeatureExtractor(cfg.opt)
        self.clf = ClassifierHandler(self.clf_path, self.opt)
        self.clf.new_classifier()

    def save(self):
        save_dict(self.opt, self.opt_path)
        self.clf.save()

    def load(self):
        self.opt = load_dict(self.opt_path)
        self.fex = FeatureExtractor(self.opt)
        self.clf = ClassifierHandler(self.clf_path)
        self.clf.load()

    def inference(self, image):
        o_height, o_width = image.shape[:2]
        x_img = self.pre_process(image)
        x_height, x_width = x_img.shape[:2]
        x_img = self.extract_features(x_img)
        x_img = np.reshape(x_img, (x_height * x_width, -1))
        y_img = self.clf.predict(x_img)
        y_img = np.reshape(y_img, (x_height, x_width))
        y_img = cv2.resize(y_img, (o_width, o_height), interpolation=cv2.INTER_NEAREST)
        return y_img

    def extract_features(self, image):
        return self.fex.extract(image)

    def pre_process(self, image):
        height, width = image.shape[:2]
        new_height = int(height / 2**self.opt["down_scale"])
        new_width = int(width / 2**self.opt["down_scale"])
        img_h = ImageHandler(image)
        img_resized = img_h.resize(height=new_height, width=new_width)
        img_h = ImageHandler(img_resized)
        img_norm = img_h.normalize()
        return img_norm

    def get_x_y(self, tag_set):
        x = []
        y = []
        print("Feature Extraction...")
        for t in tqdm(tag_set):
            x_img = t.load_x()
            x_img = self.pre_process(x_img)
            h_img, w_img = x_img.shape[:2]
            x_img = self.extract_features(x_img)
            y_img = t.load_y([h_img, w_img])

            x_img = np.reshape(x_img, (h_img * w_img, -1))
            y_img = np.reshape(y_img, h_img * w_img)

            x.append(x_img)
            y.append(y_img)
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        return x, y

    def fit(self, train_set, validation_set):
        x_train, y_train = self.get_x_y(train_set)
        x_val, y_val = self.get_x_y(validation_set)

        self.clf.fit(x_train, y_train)
        self.clf.evaluate(x_val, y_val)

