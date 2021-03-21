import cv2
import numpy as np

DEFAULT = 800

def resize(img, height=DEFAULT, allways=False):
    """Resize image to given height"""
    if (img.shape[0] > height or allways):
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))
    return img

def ratio(img, height=DEFAULT):
    """Get scale ratio"""
    return img.shape[0] / height