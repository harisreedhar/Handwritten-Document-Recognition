import os
import sys
import numpy as np
import tensorflow as tf

class CNN:
    def setupCNN(self, inputImgs, is_train):
        "create CNN layers and return output of these layers"

        cnnIn4d = tf.expand_dims(input=inputImgs, axis=3)

        # list of parameters for the layers
        kernelVals = [5, 5, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        numLayers = len(strideVals)

        # create layers
        pool = cnnIn4d  # input to first CNN layer

        for i in range(numLayers):
            kernel = tf.Variable(
                tf.random.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]],
                                           stddev=0.1))
            conv = tf.nn.conv2d(input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            conv_norm = tf.compat.v1.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool2d(input=relu, ksize=(1, poolVals[i][0], poolVals[i][1], 1),
                                    strides=(1, strideVals[i][0], strideVals[i][1], 1), padding='VALID')

        cnnOut4d = pool
        return cnnOut4d
