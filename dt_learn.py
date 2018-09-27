import numpy as numpy
import scipy as sp
from DataManipulations import *


if __name__ == "__main__":

    # read in the data
    trainFile = "heart_train.arff"
    testFile = "heart_test.arff"

    dataTrain, metaTrain = readData(trainFile)
    dataTest, metaTest = readData(testFile)

    # obtain unique features
    featureTrain = getUniqueFeatures(dataTrain, metaTrain)
    featureTest = getUniqueFeatures(dataTest, metaTest)


