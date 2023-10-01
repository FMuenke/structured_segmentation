import cv2
import numpy as np
from skimage.transform import rotate


class ChannelShift:
    def __init__(self, intensity, seed=2022):
        self.name = "ChannelShift"
        assert 1 < intensity < 255, "Set the pixel values to be shifted (1, 255)"
        self.intensity = intensity
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, img):
        height, width, ch = img.shape
        img = img.astype(np.float32)
        for i in range(ch):
            img[:, :, i] += self.rng.integers(self.intensity) * self.rng.choice([1, -1])
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)


class Stripes:
    def __init__(self, horizontal, vertical, space, width, intensity):
        self.name = "Stripes"
        self.horizontal = horizontal
        self.vertical = vertical
        self.space = space
        self.width = width
        self.intensity = intensity

    def apply(self, img):
        h, w, c = img.shape
        g_h = int(h / self.width)
        g_w = int(w / self.width)
        mask = np.zeros([g_h, g_w, c])

        if self.horizontal:
            mask[::self.space, :, :] = self.intensity
        if self.vertical:
            mask[:, ::self.space, :] = self.intensity

        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        img = mask.astype(np.float32) + img.astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)


class Blurring:
    def __init__(self, kernel=9, randomness=-1, seed=2022):
        self.name = "Blurring"
        if randomness == -1:
            randomness = kernel - 2
        assert 0 < randomness < kernel, "REQUIREMENT: 0 < randomness ({}) < kernel({})".format(randomness, kernel)
        self.kernel = kernel
        self.randomness = randomness
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, img):
        k = self.kernel + self.rng.integers(-self.randomness, self.randomness)
        img = cv2.blur(img.astype(np.float32), ksize=(k, k))
        return img.astype(np.uint8)


class NeedsMoreJPG:
    def __init__(self, percentage, randomness, seed=2022):
        self.name = "NeedsMoreJPG"
        self.percentage = percentage
        self.randomness = randomness
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, img):
        h, w = img.shape[:2]
        p = self.percentage + self.rng.integers(-self.randomness, self.randomness)
        img = cv2.resize(img, (int(w * p / 100), int(h * p / 100)), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        return img


class SaltNPepper:
    def __init__(self, max_delta, grain_size, seed=2022):
        self.name = "SaltNPepper"
        self.max_delta = max_delta
        self.grain_size = grain_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, img):
        h, w, c = img.shape
        snp_h = int(h / self.grain_size)
        snp_w = int(w / self.grain_size)
        snp = self.rng.integers(-self.max_delta, self.max_delta, size=[snp_h, snp_w, c])
        snp = cv2.resize(snp, (w, h), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.int) + snp
        return np.clip(img, 0, 255).astype(np.uint8)


def apply_noise(img):
    noise = SaltNPepper(
        max_delta=np.random.choice([4, 8, 16]),
        grain_size=np.random.choice([1, 2, 4, 8, 16])
    )
    return noise.apply(img)


def apply_blur(img):
    noise = Blurring(kernel=9, randomness=5)
    return noise.apply(img)


def apply_channel_shift(img):
    noise = ChannelShift(intensity=np.random.choice([4, 8, 16]))
    return noise.apply(img)

