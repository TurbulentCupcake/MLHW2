
import sys
import numpy as np
import scipy as sp
from DataManipulations import *
from funcs import *
import random
import matplotlib.pyplot as plt
import collections

if __name__ == '__main__':

    tSetSizes = [0.05,0.10,0.20,0.50,1]

    trainSetFile = "diabetes_train.arff"
    testSetFile = "diabetes_test.arff"
    trainData, trainMeta = readData(trainSetFile)
    testData, testMeta = readData(testSetFile)

    m = 4
    trial_accuracies = []
    for i in range(0,15):
        heart_test_accuracies = []
        for size in tSetSizes:
            count = int(size*len(trainData))
            sample_indices = random.sample(range(0, len(trainData)), count)
            sample_train = trainData[sample_indices]
            counts = collections.Counter(sample_train['class'])
            while counts[b'negative'] == counts[b'positive']:
                sample_indices = random.sample(range(0, len(trainData)), count)
                sample_train = trainData[sample_indices]
                counts = collections.Counter(sample_train['class'])
            rootnode = train(sample_train, testMeta, m)
            predictions = classify(testData, testMeta, rootnode)
            heart_test_accuracies.append(printPredictions(predictions, testData, False)/len(testData))
            # print(printPredictions(predictions, testData, False))
        trial_accuracies.append(heart_test_accuracies)

    print(np.mean(np.array(trial_accuracies), axis=0))
    trial_accuracies = np.mean(np.array(trial_accuracies), axis=0)

    plt.plot([t*100 for t in tSetSizes], [x*100 for x in trial_accuracies])
    plt.xlabel('Data Set Size')
    plt.ylabel('Percentage Accuracy')
    plt.title('diabetes_train.arff')
    plt.ylim((0,100))
    plt.xticks([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.show()











