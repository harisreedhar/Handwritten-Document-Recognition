import cv2
import numpy as np
from PIL import Image
from segmentation import Segmenter, autoPerspectiveCorrection, removeShadows, thickenImage

from spellchecker import SpellChecker
spell = SpellChecker()

from recognition import getModel, preprocess
(model, b) = getModel()

def textFromImage(path, res=1200, thick=0, shad=False, perp=False, autocorrect=True):
    image = cv2.imread(path)
    image = autoPerspectiveCorrection(image)
    image = removeShadows(image)
    image = thickenImage(image, amount=1).astype('uint8')
    segmenter = Segmenter(image, size=res)
    lines = segmenter.getLines()

    finalText = ""
    lineNums = len(lines)
    wordNums = len([word for line in lines for word in line])
    wordCount = 0
    for line in lines:
        for img in line:
            try:
                recognized, prob = recognize_with_autocorrection(img, autocorrect)
                finalText += recognized
            except Exception as e:
                print("ERROR: ", e)
                finalText += "______"
            finalText += " "
            wordCount += 1
        finalText += "\n"

    return finalText

def recognize(img):
    _img = preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (128, 32))
    _b = b(None, [_img])
    (recognized, probability) = model.inferBatch(_b, True)
    return recognized[0], probability[0]

def recognize_with_autocorrection(img, autocorrect):
    text, prob = recognize(img)
    if autocorrect:
        if prob < 0.5:
            text = spell.correction(text)
    return text, prob

if __name__ == '__main__':

    text = textFromImage("./test/test_01.jpg")

    print('____________ output ____________')
    print(text)
    print('________________________________')
