
import numpy as np
import scipy as sp
from DataManipulations import *
from funcs import *

def testGetConditionalEntropy_Numeric():
    trainFile = "heart_train.arff"
    data, meta = readData(trainFile)
    feature = 'age'
    feature_map = getUniqueFeatures(data, meta)
    minSplit, minEntropy = getConditionalEntropy_Numeric(data, meta, feature, feature_map)
    print("Entropy for Feature" + feature + " :  " + "ENTROPY = " + str(minEntropy) + "| " + "MINSPLIT = "  + str(minSplit)) 



if __name__ == "__main__":

    testGetConditionalEntropy_Numeric()