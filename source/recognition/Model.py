import os
import sys
import numpy as np
import tensorflow as tf

from . CNN import CNN
from . RNN import RNN
from . CTC import CTC

# Disable eager mode
tf.compat.v1.disable_eager_execution()

MODEL_DIR = './recognition/result/'
DUMP_DIR = './recognition/dump/'

class Model(CNN, RNN, CTC):
    "minimalistic TF model for HTR"

    # model constants
    imgSize = (128, 32)
    maxTextLen = 32

    def __init__(self, charList, mustRestore=False, dump=False):
        "init model: add CNN, RNN and CTC and initialize TF"
        self.dump = dump
        self.charList = charList
        self.mustRestore = mustRestore
        self.snapID = 0

        # Whether to use normalization over a batch or a population
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

        # input image batch
        self.inputImgs = tf.compat.v1.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

        # setup CNN, RNN and CTC
        cnnOut = self.setupCNN(self.inputImgs, self.is_train)
        rnnOut = self.setupRNN(cnnOut, charList)
        ctcOut = self.setupCTC(rnnOut, Model.maxTextLen, self.charList)

        (self.ctcIn3dTBC, self.gtTexts,
         self.seqLen, self.loss, self.savedCtcInput,
         self.lossPerElement, self.decoder, self.wbsInput) = ctcOut

        # setup optimizer to train NN
        self.batchesTrained = 0
        self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)
            #self.optimizer = tf.compat.v1.train.RMSPropOptimizer().minimize(self.loss)

        # initialize TF
        (self.sess, self.saver) = self.setupTF()

    def setupTF(self):
        "initialize TF"
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)

        sess = tf.compat.v1.Session()  # TF session

        saver = tf.compat.v1.train.Saver(max_to_keep=1)  # saver saves model to file
        modelDir = MODEL_DIR
        latestSnapshot = tf.train.latest_checkpoint(modelDir)  # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)

        # load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.compat.v1.global_variables_initializer())

        return (sess, saver)

    def toSparse(self, texts):
        "put ground truth texts into sparse tensor for ctc_loss"
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            labelStr = [self.charList.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)

    def decoderOutputToText(self, ctcOutput, batchSize):
        "extract texts from output of CTC decoder"
        labelStrs = ctcOutput
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in labelStrs]

    def trainBatch(self, batch):
        "feed a batch into the NN to train it"
        numBatchElements = len(batch.imgs)
        sparse = self.toSparse(batch.gtTexts)
        evalList = [self.optimizer, self.loss]

        feedDict = {self.inputImgs: batch.imgs,
                    self.gtTexts: sparse,
                    self.seqLen: [Model.maxTextLen] * numBatchElements,
                    self.is_train: True}

        _, lossVal = self.sess.run(evalList, feedDict)
        self.batchesTrained += 1
        return lossVal

    def dumpNNOutput(self, rnnOutput):
        "dump the output of the NN to CSV file(s)"
        dumpDir = DUMP_DIR
        if not os.path.isdir(dumpDir):
            os.mkdir(dumpDir)

        # iterate over all batch elements and create a CSV file for each one
        maxT, maxB, maxC = rnnOutput.shape
        for b in range(maxB):
            csv = ''
            for t in range(maxT):
                for c in range(maxC):
                    csv += str(rnnOutput[t, b, c]) + ';'
                csv += '\n'
            fn = dumpDir + 'rnnOutput_' + str(b) + '.csv'
            print('Write dump of NN to file: ' + fn)
            with open(fn, 'w') as f:
                f.write(csv)

    def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
        "feed a batch into the NN to recognize the texts"

        # decode, optionally save RNN output
        numBatchElements = len(batch.imgs)

        # put tensors to be evaluated into list
        evalList = []

        evalList.append(self.wbsInput)

        if self.dump or calcProbability:
            evalList.append(self.ctcIn3dTBC)

        # dict containing all tensor fed into the model
        feedDict = {self.inputImgs: batch.imgs,
                    self.seqLen: [Model.maxTextLen] * numBatchElements,
                    self.is_train: False}

        # evaluate model
        evalRes = self.sess.run(evalList, feedDict)

        # word beam search decoder: decoding is done in C++ function compute()
        decoded = self.decoder.compute(evalRes[0])

        # map labels (numbers) to character string
        texts = self.decoderOutputToText(decoded, numBatchElements)

        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calcProbability:
            sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
            ctcInput = evalRes[1]
            evalList = self.lossPerElement

            feedDict = {self.savedCtcInput: ctcInput,
                        self.gtTexts: sparse,
                        self.seqLen: [Model.maxTextLen] * numBatchElements,
                        self.is_train: False}

            lossVals = self.sess.run(evalList, feedDict)
            probs = np.exp(-lossVals)

        # dump the output of the NN to CSV file(s)
        if self.dump:
            self.dumpNNOutput(evalRes[1])

        return texts, probs

    def save(self):
        "save model to file"
        self.snapID += 1
        self.saver.save(self.sess, MODEL_DIR + 'snapshot', global_step=self.snapID)
