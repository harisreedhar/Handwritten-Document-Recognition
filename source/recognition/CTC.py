import os
import sys
import numpy as np
import tensorflow as tf

class CTC:
    def setupCTC(self, rnnOut3d, maxTextLen, charList):
        "create CTC loss and decoder and return them"
        # BxTxC -> TxBxC
        ctcIn3dTBC = tf.transpose(a=rnnOut3d, perm=[1, 0, 2])
        # ground truth text as sparse tensor
        gtTexts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]),
                                       tf.compat.v1.placeholder(tf.int32, [None]),
                                       tf.compat.v1.placeholder(tf.int64, [2]))

        # calc loss for batch
        seqLen = tf.compat.v1.placeholder(tf.int32, [None])
        loss = tf.reduce_mean(input_tensor=tf.compat.v1.nn.ctc_loss(labels=gtTexts, inputs=ctcIn3dTBC,
                                                                         sequence_length=seqLen,
                                                                         ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        savedCtcInput = tf.compat.v1.placeholder(tf.float32,
                                                      shape=[maxTextLen, None, len(charList) + 1])
        lossPerElement = tf.compat.v1.nn.ctc_loss(labels=gtTexts, inputs=savedCtcInput,
                                                       sequence_length=seqLen, ctc_merge_repeated=True)

        # best path decoding
        #self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
        chars = str().join(charList)
        wordChars = open('./recognition/result/wordCharList.txt').read().splitlines()[0]
        corpus = open('./recognition/data/corpus.txt').read()

        # decode using the "Words" mode of word beam search
        from word_beam_search import WordBeamSearch
        decoder = WordBeamSearch(50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'),
                                        wordChars.encode('utf8'))

        # the input to the decoder must have softmax already applied
        wbsInput = tf.nn.softmax(ctcIn3dTBC, axis=2)

        ctcOut = (ctcIn3dTBC, gtTexts, seqLen, loss, savedCtcInput, lossPerElement, decoder, wbsInput)
        return ctcOut
