import cv2


def resize(data, width, height, interpolation="nearest"):
    if interpolation == "nearest":
        data = cv2.resize(data, (width, height), interpolation=cv2.INTER_NEAREST)
    elif interpolation == "linear":
        data = cv2.resize(data, (width, height), interpolation=cv2.INTER_LINEAR)
    elif interpolation == "cubic":
        data = cv2.resize(data, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError("Does not know interpolation: {}".format(interpolation))
    return data
