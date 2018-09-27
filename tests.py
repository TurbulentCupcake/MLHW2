
import numpy as np
import scipy as sp
from DataManipulations import *
from funcs import *

def testGetConditionalEntropy_Numeric():
    trainFile = "heart_train.arff"
    data, meta = readData(trainFile)
    feature = 'age'
    feature_map = getUniqueFeatures(data, meta)
    print("Entropy for Feature " + age + " = " + getConditionalEntropy_Numeric(data, 
                                                                meta,
                                                                feature,
                                                                feature_map))



if __name__ == "__main__":

    testGetConditionalEntropy_Numeric()