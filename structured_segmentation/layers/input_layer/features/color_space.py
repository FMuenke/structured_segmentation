import numpy as np
import cv2


def log_geometric_mean_chromaticity(image):
    """
    This function returns the log_geometric_mean_chromaticity colorspace
    """
    image = image.astype(np.int32)
    blue = np.add(image[:, :, 0], 1)
    green = np.add(image[:, :, 1], 1)
    red = np.add(image[:, :, 2], 1)
    geometric_mean = np.cbrt(np.multiply(np.multiply(blue, green), red))
    blue_m = np.divide(blue, geometric_mean)
    green_m = np.divide(green, geometric_mean)
    red_m = np.divide(red, geometric_mean)

    blue_m = np.log(blue_m)
    green_m = np.log(green_m)
    red_m = np.log(red_m)

    return np.stack([red_m, green_m, blue_m], axis=2)

def blue_green_red(image):
    return image

def red_green_blue(image):
    blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return np.stack([red, green, blue], axis=2)

def gray(image):
    """
    This function returns the image in a gray-scale image format
    """
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return np.expand_dims(gray, axis=2)

def gray_norm(image):
    """
    This function returns the image in a gray-scale image format
    """
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255
    return np.expand_dims(gray, axis=2)

def hsv(image):
    """
    This function returns the image transformed to the HSV color-space
    H (Hue) S (Saturation) V (Value)
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def opponent(image):
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    opp_1 = np.divide((red - green), np.sqrt(2))
    opp_2 = np.divide((red + green - np.multiply(blue, 2)), np.sqrt(6))
    opp_3 = np.divide((red + green + blue), np.sqrt(3))
    return np.stack([opp_1, opp_2, opp_3], axis=2)

def rgb(image):
    blue = image[:, :, 0].astype(np.float32)
    green = image[:, :, 1].astype(np.float32)
    red = image[:, :, 2].astype(np.float32)
    red_norm = 100 * np.divide(red, (red + green + blue + 1e-5))
    green_norm = 100 * np.divide(green, (red + green + blue + 1e-5))
    blue_norm = 100 * np.divide(blue, (red + green + blue + 1e-5))
    return np.stack([red_norm, blue_norm, green_norm], axis=2)


def init_color_space(opt):
    color_options = {
        "gray": gray,
        "ngray": gray_norm,
        "hsv": hsv,
        "rgb": rgb,
        "RGB": red_green_blue,
        "opponent": opponent,
        "log_geo_mean": log_geometric_mean_chromaticity,
    }
    return color_options[opt]


def convert_to_color_space(image, opt):
    color_space = init_color_space(opt)
    return color_space(image)



class ColorSpace:
    def __init__(self, color_space):
        self.color_space = color_space
        self.func = init_color_space(self.color_space)

    def compute(self, image):
        image = np.copy(image)
        return self.func(image)
