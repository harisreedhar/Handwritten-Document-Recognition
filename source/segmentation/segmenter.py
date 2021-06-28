import cv2
import imutils
import numpy as np
from PIL import Image
from . utils import (resize, ratio, writeText,
                     thickenImage, findEdges,
                     autoPerspectiveCorrection, removeShadows)

class Segmenter:
    def __init__(self, img, size=1200):
        self.image = img
        self.size = size
        self.previewProcess = []
        self.orginalImage = self.image.copy()

    def getLines(self):
        self.image = cv2.GaussianBlur(self.image, (5, 5), 18)
        blobs = self.createBlob(self.image)
        boundingBoxes = self.createBoundingBoxes(blobs, self.image)
        self.boxLines = self.sortBoundingBoxes(boundingBoxes)
        cropped = self.cropImages(self.boxLines)
        return cropped

    def getWords(self):
        lines = self.getLines()
        return [word for line in lines for word in line]

    def createBlob(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 15)
        edgeImage = findEdges(blurred)
        _, thresholdImage = cv2.threshold(edgeImage, 50, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        self.blobs = cv2.morphologyEx(thresholdImage, cv2.MORPH_CLOSE, kernel)

        self.previewProcess.append([blurred])
        self.previewProcess.append([edgeImage])
        self.previewProcess.append([self.blobs])

        return self.blobs

    def createBoundingBoxes(self, blobs, image):
        size = self.size
        blobImage = resize(blobs, size)
        kernel = np.ones((5, 30), np.uint16)
        imageDilation = cv2.dilate(blobImage, kernel, iterations=1)
        _, cnt, hierarchy = cv2.findContours(np.copy(blobImage), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = np.zeros(blobImage.shape, dtype='uint8') #resize(image.copy(), size)
        cv2.drawContours(contours, cnt, -1, (255,0,0), 3)
        self.previewProcess.append([contours])

        index = 0
        boxes = []
        while(index >= 0):
            x, y, w, h = cv2.boundingRect(cnt[index])
            cv2.drawContours(imageDilation, cnt, index, (255, 255, 255), cv2.FILLED)
            maskROI = imageDilation[y:y+h, x:x+w]
            r = cv2.countNonZero(maskROI) / (w * h)

            limit = (r > 0.1
                    and size > w > 10
                    and size > h > 10
                    and (h / w) < 3
                    and (w / h) < 10
                    and (60 // h) * w < 1000)
            if limit:
                boxes += [[x, y, w, h]]
            index = hierarchy[0][index][0]

        blobImage = cv2.cvtColor(blobImage, cv2.COLOR_GRAY2RGB)
        boundingBoxes = np.array([0,0,0,0])

        for (x, y, w, h) in boxes:
            boundingBoxes = np.vstack((boundingBoxes, np.array([x, y, x+w, y+h])))

        boxes = boundingBoxes.dot(ratio(image, blobImage.shape[0])).astype(np.int64)
        return boxes[1:]

    def sortBoundingBoxes(self, boxes):
        boxes = boxes[boxes[:, 1].argsort(kind='mergesort')]

        meanHeight = np.mean(boxes[:, 3] - boxes[:, 1])
        currentBox = boxes[0][1]
        lines = []
        tmpLine = []

        for box in boxes:
            if box[1] > currentBox + meanHeight:
                lines.append(tmpLine)
                if box[1] > currentBox + (2.5 * meanHeight):
                    lines.append([])
                tmpLine = [box]
                currentBox = box[1]
                continue
            tmpLine.append(box)
        lines.append(tmpLine)

        for line in lines:
            line.sort(key=lambda box: box[0])

        return lines

    def cropImages(self, lines):
        imgs = []
        for line in lines:
            text = self.image.copy()
            lineTexts = []
            for (x1, y1, x2, y2) in line:
                lineTexts.append(text[y1:y2, x1:x2])
            imgs.append(lineTexts)
        return imgs

    def drawTextOnBoundingBoxes(self, labels):
        sortedBoxes = [box for boxLine in self.boxLines for box in boxLine]
        shape = self.orginalImage.shape
        boundingBoxImage = self.orginalImage.copy()
        crop = self.orginalImage.copy()
        mask = np.zeros((shape[0], shape[1]), dtype='uint8')

        for i, (x1,y1,x2,y2) in enumerate(sortedBoxes):
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            cv2.rectangle(boundingBoxImage, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 0, 0), -1)

        crop = cv2.bitwise_and(crop, crop, mask = mask)
        finalPreview = crop.copy()

        for i, (x1,y1,x2,y2) in enumerate(sortedBoxes):
            cv2.putText(finalPreview, labels[i], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200,200,0), 2)

        self.previewProcess.append([boundingBoxImage])
        self.previewProcess.append([crop])
        self.previewProcess.append([finalPreview])
