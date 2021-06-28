import cv2
import imutils
import numpy as np
from imutils import perspective
from PIL import Image

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

def thickenImage(img, amount=1, kernelSize = 3):
    if amount == 0:
        return img
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    if amount < 0:
        thinned = cv2.dilate(img, kernel, iterations = -amount)
        return thinned.astype('uint8')
    thickened = cv2.erode(img, kernel, iterations = amount)
    return thickened.astype('uint8')

def sobelDetect(channel):
    """Sobel operator for edge detection"""

    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    # Find magnitude sqrt(sobelX**2 + sobelY**2)
    sobel = np.hypot(sobelX, sobelY)
    # Clamp values > 255
    sobel[sobel > 255] = 255
    return np.uint8(sobel)

def findEdges(image):
    """Find edges using sobel operator on each layer(RGB)"""

    return np.max(np.array([sobelDetect(image[:,:,0]),
                            sobelDetect(image[:,:,1]),
                            sobelDetect(image[:,:,2])]), axis=0)

# https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv
def removeShadows(image, norm=False):
    rgb_planes = cv2.split(image)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane,kernel=(1,1),iterations=1)
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        ret, norm_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
        cv2.normalize(norm_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm


# https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
def autoPerspectiveCorrection(image):
    ratio = image.shape[0] / 500
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    screenCnt = 0

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    warped = perspective.four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    return warped

def stackImages(images, mode='h'):
    images, w, h = pad_images_to_same_size(images)
    imstack = cv2.resize(images[0],(w,h))
    for im in images[1:]:
        if mode == 'v':
            imstack = np.vstack((imstack,im))
        if mode == 'h':
            imstack = np.hstack((imstack,im))
    return imstack

def pad_images_to_same_size(images):
    width_max = 0
    height_max = 0
    for img in images:
        h, w = img.shape[:2]
        width_max = max(width_max, w)
        height_max = max(height_max, h)

    images_padded = []
    for img in images:
        h, w = img.shape[:2]
        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        assert img_padded.shape[:2] == (height_max, width_max)
        images_padded.append(img_padded)

    return images_padded, width_max, height_max
