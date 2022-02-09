import numpy as np

from data_structure.image_container import ImageContainer


class ColorSpace:
    def __init__(self, color_space):
        self.color_space = color_space

    def _prepare_image(self, image):
        img_h = ImageContainer(image)
        return img_h.prepare_image_for_processing(self.color_space)

    def compute(self, image):
        list_of_targets = self._prepare_image(image)
        tensor = []
        for t in list_of_targets:
            t = np.expand_dims(t, axis=2)
            tensor.append(t)
        return np.concatenate(tensor, axis=2)
