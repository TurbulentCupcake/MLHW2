
import numpy as np
import scipy as sp
import collections as collections
import math
import sys



def getEntropyParent(data, meta):
    """
        This function computes entropy for a given feature
        Note: This does not compute the conditional entropy,
        see getConditionalEntropy for this case
    """

    counts = dict(collections.Counter(data['class']))
    counts_total = sum(counts.values())
    entropy = 0
    for key in list(counts.keys()):
        entropy += -(counts[key]/counts_total)*(math.log((counts[key]/counts_total),2))
    return entropy
 

def getConditionalEntropy_Nominal(data, meta, feature, feature_map):
    """
        This function is used to obtain the entropy for features with nominal values
        It computes the conditional entropy for each nominal value, multiplies by
        the fraction of occurances in the data for which that value is present
        and sums this value and returns it back to the user

        Note: This returns not the individual entropies of each feature with respect 
        to its nominal value, but weighted entropy of each feature
    """

    # check if the feature passed by the user is a nomianl feature
    assert meta.types()[meta.names().index(feature)] == 'nominal'

    nominal_vals = feature_map[feature]

    total_entropy = 0
    # iterate through each possible nominal value
    for val in nominal_vals:

        cond_entropy = 0 # store conditional entropy
        counts = [] # list that keeps track of positive and negatives
        for i in range(len(data)):
            if data[feature][i] == val:
                counts.append(data['class'][i])
        
        counts = dict(collections.Counter(counts))
        counts_total = sum(counts.values()) # get the total number of occurances for this nominal value

        # compute conditional entropy
        for key in list(counts.keys()):
            cond_entropy += -(counts[key]/counts_total)*(math.log((counts[key]/counts_total),2))
        
        # add it to the total net conditional entropy of this feature
        total_entropy += (counts_total/len(data))*cond_entropy
    
    return total_entropy 


def getConditionalEntropy_Numeric(data, meta, feature, feature_map):
    """
        This function computes the entropy for numerical category.
        Our goal here is to find the candidate split
        that gives us the lowest entropy and return the weighted entropy
        of that value
    """
    # check if the feature passed by the user is a numeric feature
    assert meta.types()[meta.names().index(feature)] == 'numeric'
    
    # compute candidate splits 
    
    numeric_vals = list(data[feature]) # obtain values for that feature    
    numeric_vals.sort()  # sort the values for this feature
    candidate_splits = [] # create a structure to hold all candditate splitds
    for i in range(len(numeric_vals)-1):
        candidate_splits.append((numeric_vals[i]+numeric_vals[i+1])/2)
    
    # for each candidate split, evaluate entropy and use the candidate split
    # that gives you the lowest entropy

    minSplit = sys.maxsize
    minEntropy = sys.maxsize
    for i, split in enumerate(candidate_splits):

        # print("Split = ", split, "| Iteration = ", i)
        cond_entropy_under_threshold = 0
        cond_entropy_above_threshold = 0
        counts_under_candidate_split = []
        counts_above_candidate_split = []

        # iterate through the data and obtain values for class
        # for numeric values less than and above a given value
        for i in range(len(data)):
            if data[feature][i] < split:
                counts_under_candidate_split.append(data['class'][i])
            else:
                counts_above_candidate_split.append(data['class'][i])
        
        # get counts for each branch 
        counts_under_candidate_split = dict(collections.Counter(counts_under_candidate_split))
        counts_above_candidate_split = dict(collections.Counter(counts_above_candidate_split))

        # get the total counts
        counts_under_candidate_split_total = 0
        counts_above_candidate_split_total = 0

        for key in counts_under_candidate_split.keys():
            counts_under_candidate_split_total += counts_under_candidate_split[key]
        
        for key in counts_above_candidate_split.keys():
            counts_above_candidate_split_total += counts_above_candidate_split[key]

        # obtain conditional entropy for both branches
        for key in list(counts_under_candidate_split.keys()):
            cond_entropy_under_threshold += -(counts_under_candidate_split[key]/counts_under_candidate_split_total)*(math.log((counts_under_candidate_split[key]/counts_under_candidate_split_total), 2))
        
        # print("Cond Entropy Under Threshold = ", cond_entropy_under_threshold)
        # print("Total count under threshold = ", counts_under_candidate_split_total)

        for key in list(counts_above_candidate_split.keys()):
            cond_entropy_above_threshold += -(counts_above_candidate_split[key]/counts_above_candidate_split_total)*(math.log((counts_above_candidate_split[key]/counts_above_candidate_split_total),2))
        
        # print("Cond Entropy Above Threshold = ", cond_entropy_above_threshold)
        # print("Total count above threshold = ", counts_above_candidate_split_total)

        # compute the weighted entropy
        temp_total_weighted_entropy =  (counts_under_candidate_split_total/len(data))*cond_entropy_under_threshold + (counts_above_candidate_split_total/len(data))*cond_entropy_above_threshold
        # print("Entropy = ", temp_total_weighted_entropy)
        # check if the new computed weighted entropy is lower than than the current
        if temp_total_weighted_entropy < minEntropy:
            minEntropy = temp_total_weighted_entropy
            minSplit = split
    
    return minSplit, minEntropy
        


        






                













    



def getInfoGain(data, type, feature):
    """
        Function to compute information gain for a given numeric feature.

    """

    # Step 1
    # For the selected feature, find the number of labels that have
    # positive and the number that have negative.
    # Using this compute the entropy for the feature itself

    # Step 2
    # This is the important part. Here, we have to compute the
    # entropy for all the branches. In order to do this,
    # first figure out what the nominal value for each
    # feature is (you can get this through the getUniqueFeature function)
    # For each of these nominal values, find out which values
    # are positive and negative and compute the conditional entropy

    # Step 3
    # Finally, just subtract the value of step 3 from step 2
    # Return answer
