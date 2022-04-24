import numpy as np
import random
import cv2
from structured_classifier.augmentation.aug_methods import *


class Augmentations:
    def __init__(self, salt_n_pepper, color_shift, blurring):
        self.salt_n_pepper = salt_n_pepper
        self.color_shift = color_shift
        self.blurring = blurring

    def apply(self, img):
        if 0 == np.random.randint(5) and self.salt_n_pepper:
            img = apply_noise(img)

        if 0 == np.random.randint(5) and self.color_shift:
            img = apply_channel_shift(img)

        if 0 == np.random.randint(5) and self.blurring:
            img = apply_blur(img)
        return img


def augment_data_set(tags, augmentations, multiplier):
    augmented_tags = []
    print("Augmenting Data Set...")
    for t in tags:
        augmented_tags.append(t)
        for i in range(multiplier):
            t_aug = t.create_augmented_tag(augmentations)
            augmented_tags.append(t_aug)
    print("Data Set Augmented.")
    random.shuffle(augmented_tags)
    return augmented_tags
