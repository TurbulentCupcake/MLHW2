
import numpy as np
import scipy as sp
import collections as collections
import math



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
    
    numeric_vals = data[feature] # obtain values for that feature    
    numeric_vals.sort()  # sort the values for this feature
    candidate_splits = [] # create a structure to hold all candditate splitds
    for i in range(len(numeric_vals)-1):
        candidate_splits.append((numeric_vals[i]+numeric_vals[i+1])/2)
    
    # for each candidate split, evaluate entropy and use the candidate split
    # that gives you the lowest entropy

    for i, split in enumerate(candidate_splits):
        








    



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
