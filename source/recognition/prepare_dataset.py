import os
import cv2
import lmdb
import pickle
import random
import numpy as np
from path import Path
from . utils import preprocess

CURR_WORKDIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR  = CURR_WORKDIR + '/dataset'
IAM_DIR      = DATASET_DIR + '/IAM/img'
LMDB_PATH    = Path(DATASET_DIR + '/IAM/lmdb')
CUSTOM_DIR   = DATASET_DIR + '/CUSTOM/'
CUSTOM_IMGS  = CUSTOM_DIR + 'ht_01'

def createLMDB():
    """For fast access when training on GPU"""

    if not (LMDB_PATH).exists():
        # 2GB is enough for IAM dataset
        env = lmdb.open(str(LMDB_PATH), map_size=1024 * 1024 * 1024 * 2)

        # go over all png files
        fn_imgs = list((Path(IAM_DIR)).walkfiles('*.png'))
        fn_imgs2 = list((Path(CUSTOM_IMGS)).walkfiles('*.png'))

        # and put the imgs into lmdb as pickled grayscale imgs
        with env.begin(write=True) as txn:

            for i, fn_img in enumerate(fn_imgs):
                print(i, len(fn_imgs))
                img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
                basename = fn_img.basename()
                txn.put(basename.encode("ascii"), pickle.dumps(img))

            for j, fn_img2 in enumerate(fn_imgs2):
                print(j, len(fn_imgs2))
                img = cv2.imread(fn_img2, cv2.IMREAD_GRAYSCALE)
                basename = fn_img2.basename()
                txn.put(basename.encode("ascii"), pickle.dumps(img))

        env.close()
    else:
        print("Skip LMDB creation")


class Sample:
    "sample from the dataset"

    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath

class Batch:
    "batch containing images and ground truth texts"

    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts

class DataLoaderIAM:
    "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database"

    def __init__(self, data_dir, batchSize, imgSize, maxTextLen, fast=True):
        "loader for dataset at given location, preprocess images and text according to parameters"

        assert data_dir.exists()

        self.fast = fast
        if fast:
            createLMDB()
            self.env = lmdb.open(str(LMDB_PATH), readonly=True)

        self.dataAugmentation = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []

        f = open(data_dir / 'gt/words.txt')
        chars = set()
        # known broken images in IAM dataset
        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']

        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 9

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileNameSplit = lineSplit[0].split('-')
            fileName = data_dir / 'img' / fileNameSplit[0] / f'{fileNameSplit[0]}-{fileNameSplit[1]}' / lineSplit[0] + '.png'

            if lineSplit[0] in bad_samples_reference:
                print('Ignoring known broken image:', fileName)
                continue

            # GT text are columns starting at 9
            gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
            chars = chars.union(set(list(gtText)))

            # put sample into list
            self.samples.append(Sample(gtText, fileName))

        # open custom dataset label
        f2 = open(CUSTOM_DIR + 'ht_01_label.txt')

        # append custom dataset to samples
        for i, label in enumerate(f2):
            label = label.strip('\n')
            chars = chars.union(set(list(label)))
            customPath = CUSTOM_DIR + f'ht_01/ht_{i}.png'

            self.samples.append(Sample(label, customPath))

        # split into training and validation set: 95% - 5%
        splitIdx = int(0.95 * len(self.samples))
        self.trainSamples = self.samples[:splitIdx]
        self.validationSamples = self.samples[splitIdx:]

        # put words into lists
        self.trainWords = [x.gtText for x in self.trainSamples]
        self.validationWords = [x.gtText for x in self.validationSamples]

        # start with train set
        self.trainSet()

        # list of all chars in dataset
        self.charList = sorted(list(chars))

    def truncateLabel(self, text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text

    def trainSet(self):
        "switch to randomly chosen subset of training set"

        self.dataAugmentation = True
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples
        self.currSet = 'train'

    def validationSet(self):
        "switch to validation set"

        self.dataAugmentation = False
        self.currIdx = 0
        self.samples = self.validationSamples
        self.currSet = 'val'

    def getIteratorInfo(self):
        "current batch index and overall number of batches"

        if self.currSet == 'train':
            # train set: only full-sized batches
            numBatches = int(np.floor(len(self.samples) / self.batchSize))
        else:
            # val set: allow last batch to be smaller
            numBatches = int(np.ceil(len(self.samples) / self.batchSize))
        currBatch = self.currIdx // self.batchSize + 1
        return currBatch, numBatches

    def hasNext(self):
        "iterator"

        if self.currSet == 'train':
            # train set: only full-sized batches
            return self.currIdx + self.batchSize <= len(self.samples)
        else:
            # val set: allow last batch to be smaller
            return self.currIdx < len(self.samples)

    def getNext(self):
        "iterator"

        batchRange = range(self.currIdx, min(self.currIdx + self.batchSize, len(self.samples)))
        gtTexts = [self.samples[i].gtText for i in batchRange]

        imgs = []
        for i in batchRange:
            if self.fast:
                with self.env.begin() as txn:
                    basename = Path(self.samples[i].filePath).basename()
                    data = txn.get(basename.encode("ascii"))
                    img = pickle.loads(data)
            else:
                img = cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE)

            imgs.append(preprocess(img, self.imgSize, self.dataAugmentation))

        self.currIdx += self.batchSize

        return Batch(gtTexts, imgs)
