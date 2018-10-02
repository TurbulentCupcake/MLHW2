
import sys
import numpy as np
import scipy as sp
from DataManipulations import *
from funcs import *

if __name__ == '__main__':
    # print(sys.argv)
    trainSetFile = sys.argv[1]
    testSetFile = sys.argv[2]
    data, meta = readData(trainSetFile)
    testData, testMeta = readData(testSetFile)
    feature_map = getUniqueFeatures(data, meta)
    m = int(sys.argv[3])

    rootnode = train(data, meta, m)
    predictions = classify(testData, testMeta, rootnode)
    print('<Predictions for the Test Set Instances>')
    printPredictions(predictions, testData)



