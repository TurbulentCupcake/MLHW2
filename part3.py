
import sys
import numpy as np
import scipy as sp
from DataManipulations import *
from funcs import *
import random
import matplotlib.pyplot as plt
import collections


if __name__ == "__main__":

    trainSetFile = "diabetes_train.arff"
    testSetFile = "diabetes_test.arff"
    trainData, trainMeta = readData(trainSetFile)
    testData, testMeta = readData(testSetFile)

    accuracies = []
    for m in [2,5,10, 20]:
        rootnode = train(trainData, trainMeta, m)
        predictions = classify(testData, testMeta, rootnode)
        accuracies.append(printPredictions(predictions, testData, False) / len(testData))

    print(accuracies)
    plt.plot([1,2,3,4], [x*100 for x in accuracies])
    plt.xlabel('Data Set Size')
    plt.ylabel('Percentage Accuracy')
    plt.title(trainSetFile)
    plt.ylim((0,100))
    plt.xticks([1,2,3,4], labels=[2,5,10,20])
    plt.show()


