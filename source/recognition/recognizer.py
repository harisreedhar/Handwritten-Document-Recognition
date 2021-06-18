import argparse
import json

import os
import cv2
import editdistance
from path import Path

from . prepare_dataset import DataLoaderIAM, Batch
from . Model import Model, DecoderType
from . utils import preprocess

import timeit
start = timeit.default_timer()

DATASET_DIR = Path("/home/hari/Documents/s6Project_New/recognition/dataset/")
CURR_WORKDIR = os.path.dirname(os.path.realpath(__file__))

# set wordbeam search as default decoder
decoderType = DecoderType.WordBeamSearch

class FilePaths:
    "filenames and paths to data"
    fnCharList = CURR_WORKDIR + '/model/charList.txt'
    fnSummary = CURR_WORKDIR + '/model/summary.json'
    fnInfer = CURR_WORKDIR + '/data/test.png'
    fnCorpus = CURR_WORKDIR + '/data/corpus.txt'


def write_summary(charErrorRates, wordAccuracies):
    with open(FilePaths.fnSummary, 'w') as f:
        json.dump({'charErrorRates': charErrorRates, 'wordAccuracies': wordAccuracies}, f)


def train(model, loader):
    "train NN"
    epoch = 0  # number of training epochs since start
    summaryCharErrorRates = []
    summaryWordAccuracies = []
    bestCharErrorRate = float('inf')  # best valdiation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occured
    earlyStopping = 25  # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print(f'Epoch: {epoch} Batch: {iterInfo[0]}/{iterInfo[1]} Loss: {loss}')

        # validate
        charErrorRate, wordAccuracy = validate(model, loader)

        # write summary
        summaryCharErrorRates.append(charErrorRate)
        summaryWordAccuracies.append(wordAccuracy)
        write_summary(summaryCharErrorRates, summaryWordAccuracies)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {charErrorRate * 100.0}%')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print(f'No more improvement since {earlyStopping} epochs. Training stopped.')
            break


def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print(f'Batch: {iterInfo[0]} / {iterInfo[1]}')
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print(f'Character error rate: {charErrorRate * 100.0}%. Word accuracy: {wordAccuracy * 100.0}%.')
    print(f'time:{timeit.default_timer()-start}')
    return charErrorRate, wordAccuracy


def infer(model, fnImg):
    "recognize text in image provided by file path"
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('___________ Output ___________')
    print('')
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')
    print('______________________________')

def startTrain(batchSize=200):
    loader = DataLoaderIAM(DATASET_DIR, batchSize, Model.imgSize, Model.maxTextLen, True)
    open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
    open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))
    model = Model(loader.charList, decoderType)
    train(model, loader)

def recognizeText(imagePath):
    model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=False)
    infer(model, imagePath)

def getModel():
    model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=False)
    return model, Batch

if __name__ == '__main__':
    recognizeText(FilePaths.fnInfer)
