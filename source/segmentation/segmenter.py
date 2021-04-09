import cv2
import numpy as np
from PIL import Image
from utils import resize, ratio, writeText

class DetectWords:
    debugImages = []

    def detectLines(self, img):
        """Return list of lines each line contains bounding box for words"""

        boxes = self.detectText(img)
        lines = self.sortTexts(boxes)
        return lines

    def detectText(self, img):
        """Return bounding boxes around words"""

        # Blur to eliminate high frequency noise
        blurred = cv2.GaussianBlur(img, (5, 5), 18)
        # Find edges from blurred image
        edgeImage = self.edgeDetect(blurred)
        # Find threshold image
        _, edgeImage = cv2.threshold(edgeImage, 50, 255, cv2.THRESH_BINARY)
        # Fill edges
        filledImage = cv2.morphologyEx(edgeImage, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        self.debugImages.append([blurred, "1_blurred"])
        self.debugImages.append([edgeImage, "2_edgeImage"])
        self.debugImages.append([filledImage, "3_filledImage"])

        return self.getBoundingBoxes(filledImage, img)

    def getBoundingBoxes(self, img, image, size=2000):
        """Find contours and return bounding boxes"""

        orginal = resize(image, size)
        small = resize(img, size)

        kernel = np.ones((5, 100), np.uint16)
        imageDilation = cv2.dilate(small, kernel, iterations=1)

        # Find contours
        _, cnt, hierarchy = cv2.findContours(np.copy(small), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        index = 0
        boxes = []

        while(index >= 0):
            # Get bounding rectangle around each contour
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

        small = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
        boundingBoxes = np.array([0,0,0,0])

        # Stack bounding boxes
        for (x, y, w, h) in boxes:
            # draw rectangle over original image
            cv2.rectangle(orginal, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # draw rectangle over small image
            cv2.rectangle(small, (x, y), (x+w, y+h), (255, 255, 0), 2)
            # vertically stack
            boundingBoxes = np.vstack((boundingBoxes, np.array([x, y, x+w, y+h])))

        # transform bounding boxes to fit orginal image
        boxes = boundingBoxes.dot(ratio(image, small.shape[0])).astype(np.int64)

        self.debugImages.append([orginal, "5_original_BB"])
        self.debugImages.append([small, "4_small_BB"])

        return boxes[1:]

    def edgeDetect(self, img):
        """Find edges using sobel operator on each layer(RGB)"""

        return np.max(np.array([self.sobelDetect(img[:,:,0]),
                                self.sobelDetect(img[:,:,1]),
                                self.sobelDetect(img[:,:,2])]), axis=0)

    def sobelDetect(self, channel):
        """Sobel operator for edge detection"""

        # Horizontal sobel
        sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
        # Vertical sobel
        sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
        # Find magnitude sqrt(sobelX**2 + sobelY**2)
        sobel = np.hypot(sobelX, sobelY)
        # Clamp values > 255
        sobel[sobel > 255] = 255
        return np.uint8(sobel)

    def sortTexts(self, boxes):
        """Sort bounding boxes and collect to lines"""

        # average height of bounding boxes
        meanHeight = sum([y2 - y1 for _, y1, _, y2 in boxes]) / len(boxes)
        # sort boxes based on y
        boxes.view('i8,i8,i8,i8').sort(order=['f1'], axis=0)
        currentLines = boxes[0][1]
        lines = []
        tmpLine = []

        for box in boxes:
            # if box > currentLines + meanHeight: create new line
            # else: add to current line
            if box[1] > currentLines + meanHeight:
                lines.append(tmpLine)
                tmpLine = [box]
                currentLines = box[1]
                continue
            tmpLine.append(box)
        lines.append(tmpLine)

        # sort words inside line based on x
        for line in lines:
            line.sort(key=lambda box: box[0])

        return lines

class Segmenter(DetectWords):
    """Segment image into sorted words line by line"""

    def __init__(self, img):
        self.image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        self.lines = self.detectLines(self.image)

    def getLineTexts(self):
        """Return list of cropped images"""

        imgs = []
        for line in self.lines:
            text = self.image.copy()
            lineTexts = []
            for (x1, y1, x2, y2) in line:
                lineTexts.append(text[y1:y2, x1:x2])
            imgs.append(lineTexts)
        return imgs

    def save(self, path='./result/'):
        """Save segmented image to path"""

        imgs = self.getLineTexts()
        i = 0
        for line in imgs:
            for word in line:
                image = Image.fromarray(word)
                image.save(path + f"segment_{i}.png")
                print(f"Segment word {i} done")
                i += 1
        return

    def saveDebugImages(self):
        """For visualizing whole process"""

        for d in self.debugImages:
            img = Image.fromarray(d[0])
            img.save(d[1] + ".png")

if __name__ == '__main__':
    s = Segmenter('test.jpg')
    s.save()
    s.saveDebugImages()
