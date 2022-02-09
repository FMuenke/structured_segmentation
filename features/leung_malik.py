import numpy as np

from data_structure.matrix_container import MatrixContainer
from data_structure.image_container import ImageContainer


class LeungMalik:
    """
    generate LM filters for textons
    """

    def __init__(self, color_space="gray"):
        self.color_space = color_space
        # self.resolution = 64
        self.lm_filters = self._make_lm_filters()

    def _build_feature_tensor(self, image):
        feature_maps = []
        for f_kernel_idx in range(self.lm_filters.shape[2]):
            mat_h = MatrixContainer(image)
            feature_maps.append(mat_h.apply_convolution(self.lm_filters[:, :, f_kernel_idx]))
        return np.stack(feature_maps, axis=2)

    def _compute(self, channels):
        tensors = []
        for c in channels:
            f_tensor = self._build_feature_tensor(c)
            tensors.append(f_tensor)
        return tensors

    def compute(self, image):
        img_h = ImageContainer(image)
        channels = img_h.prepare_image_for_processing(self.color_space)
        feature_tensors = self._compute(channels)
        return np.concatenate(feature_tensors, axis=2)

    def _gaussian_1d(self, sigma, mean, x, ord):
        """
        return gaussian differentiation for 1st and 2nd order deravatives
        """
        x = np.array(x)
        x_ = x - mean
        var = sigma ** 2
        # Gaussian Function
        g1 = (1 / np.sqrt(2 * np.pi * var)) * (np.exp((-1 * x_ * x_) / (2 * var)))

        if ord == 0:
            g = g1
            return g  # gaussian function
        elif ord == 1:
            g = -g1 * (x_ / var)
            return g  # 1st order differentiation
        else:
            g = g1 * (((x_ * x_) - var) / (var ** 2))
            return g  # 2nd order differentiation

    def _gaussian_2d(self, sup, scales):
        var = scales * scales
        shape = (sup, sup)
        n, m = [(i - 1) / 2 for i in shape]
        x, y = np.ogrid[-m:m + 1, -n:n + 1]
        g = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x * x + y * y) / (2 * var))
        return g

    def _log_2d(self, sup, scales):
        var = scales * scales
        shape = (sup, sup)
        n, m = [(i - 1) / 2 for i in shape]
        x, y = np.ogrid[-m:m + 1, -n:n + 1]
        g = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x * x + y * y) / (2 * var))
        h = g * ((x * x + y * y) - var) / (var ** 2)
        return h

    def _make_filter(self, scale, phasex, phasey, pts, sup):

        gx = self._gaussian_1d(3 * scale, 0, pts[0, ...], phasex)
        gy = self._gaussian_1d(scale, 0, pts[1, ...], phasey)

        image = gx * gy

        image = np.reshape(image, (sup, sup))
        return image

    def _make_lm_filters(self):
        sup = 49
        scalex = np.sqrt(2) * np.array([1, 2, 3])
        norient = 6
        nrotinv = 12

        nbar = len(scalex) * norient
        nedge = len(scalex) * norient
        nf = nbar + nedge + nrotinv
        F = np.zeros([sup, sup, nf])
        hsup = (sup - 1) / 2

        x = [np.arange(-hsup, hsup + 1)]
        y = [np.arange(-hsup, hsup + 1)]

        [x, y] = np.meshgrid(x, y)

        orgpts = [x.flatten(), y.flatten()]
        orgpts = np.array(orgpts)

        count = 0
        for scale in range(len(scalex)):
            for orient in range(norient):
                angle = (np.pi * orient) / norient
                c = np.cos(angle)
                s = np.sin(angle)
                rotpts = [[c + 0, -s + 0], [s + 0, c + 0]]
                rotpts = np.array(rotpts)
                rotpts = np.dot(rotpts, orgpts)
                F[:, :, count] = self._make_filter(scalex[scale], 0, 1, rotpts, sup)
                F[:, :, count + nedge] = self._make_filter(scalex[scale], 0, 2, rotpts, sup)
                count = count + 1

        count = nbar + nedge
        scales = np.sqrt(2) * np.array([1, 2, 3, 4])

        for i in range(len(scales)):
            F[:, :, count] = self._gaussian_2d(sup, scales[i])
            count = count + 1

        for i in range(len(scales)):
            F[:, :, count] = self._log_2d(sup, scales[i])
            count = count + 1

        for i in range(len(scales)):
            F[:, :, count] = self._log_2d(sup, 3 * scales[i])
            count = count + 1

        return F
