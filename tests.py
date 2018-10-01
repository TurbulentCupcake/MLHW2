
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
        
def testInfoGain():
    trainFile = "heart_train.arff"
    data, meta = readData(trainFile)
    feature_map = getUniqueFeatures(data, meta)

    for feature, featureType in zip(meta.names(), meta.types()):
        if featureType == 'nominal':
            infogain = getInfoGain(data, meta, feature)
            print("Info gain for feature " + feature + " = " + str(infogain) )
        else:
            split, infogain = getInfoGain(data, meta, feature)
            print("Info gain for feature " + feature + " = " + str(infogain) + " | split = " + str(split))  

def testTrain():

    trainFile = "diabetes_train.arff"
    data, meta = readData(trainFile)
    feature_map = getUniqueFeatures(data, meta)
    m = 2

    rootnode = train(data, meta, m)    



if __name__ == "__main__":

    # testGetConditionalEntropy_Numeric()
    # testGetConditionalEntropy_Nominal()
    # testInfoGain()
    testTrain()