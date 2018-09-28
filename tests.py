
import numpy as np
import scipy as sp
from DataManipulations import *
from funcs import *

def testGetConditionalEntropy_Numeric():
    trainFile = "heart_train.arff"
    data, meta = readData(trainFile)
    feature_map = getUniqueFeatures(data, meta)

    for feature, featureType in zip(meta.names(), meta.types()):
        if featureType == 'numeric':        
            minSplit, minEntropy = getConditionalEntropy_Numeric(data, meta, feature, feature_map)
            print("Entropy for Feature " + feature + " :  " + "ENTROPY = " + str(minEntropy) + "| " + "MINSPLIT = "  + str(minSplit)) 

def testGetConditionalEntropy_Nominal():
    trainFile = "heart_train.arff"
    data, meta = readData(trainFile)
    feature_map = getUniqueFeatures(data, meta)
    
    for feature, featureType in zip(meta.names(), meta.types()):
        if featureType == 'nominal':
            minEntropy = getConditionalEntropy_Nominal(data, meta, feature, feature_map)
            print("Entropy for Feature " + feature + " : " + " ENTROPY = " + str(minEntropy))



if __name__ == "__main__":

    # testGetConditionalEntropy_Numeric()
    testGetConditionalEntropy_Nominal()