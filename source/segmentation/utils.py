import cv2
import numpy as np

SMALL_HEIGHT = 800

def writeText(img, text="", location=(10,10), size=1, color=(0, 0, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, text, location, font, size, color, 2, cv2.LINE_AA)
    return img

def resize(img, height=SMALL_HEIGHT, allways=False):
    if (img.shape[0] > height or allways):
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))
    return img

def ratio(img, height=SMALL_HEIGHT):
    return img.shape[0] / height
