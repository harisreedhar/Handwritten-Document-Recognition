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
    pass
