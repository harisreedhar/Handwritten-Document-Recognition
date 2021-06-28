import unittest
import cv2

TEST_IMAGE_01 = './test/test_01.jpg'
TEST_IMAGE_02 = './test/test_02.jpg'
TEST_IMAGE_03 = './test/test_03.jpg'

class Test_HWDR(unittest.TestCase):

    def test_readImage(self):
        image = cv2.imread(TEST_IMAGE_01)
        self.assertIsNotNone(image)

    def test_perspectiveCorrection(self):
        from segmentation import autoPerspectiveCorrection
        image = autoPerspectiveCorrection(cv2.imread(TEST_IMAGE_01))
        self.assertIsNotNone(image)

    def test_shadowRemoval(self):
        from segmentation import removeShadows
        image = removeShadows(cv2.imread(TEST_IMAGE_02))
        self.assertIsNotNone(image)

    def test_thickenImage(self):
        from segmentation import thickenImage
        image = thickenImage(cv2.imread(TEST_IMAGE_03), amount=1)
        self.assertIsNotNone(image)

    def test_wordSegmentation(self):
        from segmentation import Segmenter
        s = Segmenter(cv2.imread(TEST_IMAGE_03), size=1200)
        lines = s.getLines()
        words = s.getWords()
        self.assertTrue(len(lines) > 0)
        self.assertTrue(len(words) > 0)

    def test_model(self):
        from segmentation import Segmenter
        from recognition import getModel, preprocess
        (Model, Infe) = getModel()
        s = Segmenter(cv2.imread(TEST_IMAGE_03), size=1200)
        words = s.getWords()
        output = []
        probs = []
        for word in words:
            word = preprocess(cv2.cvtColor(word, cv2.COLOR_BGR2GRAY), (128, 32))
            _b = Infe(None, [word])
            (recognized, probability) = Model.inferBatch(_b, True)
            output.append(recognized[0])
            probs.append(probability[0])
        self.assertTrue(len(output) > 0)
        self.assertTrue(len(probs) > 0)

if __name__ == '__main__':
    unittest.main()
