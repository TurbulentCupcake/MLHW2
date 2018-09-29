
import numpy as np
import scipy as sp
import collections as collections
import math
import sys
from DataManipulations import *
from DTNode import * 

SPLIT_CONSENSUS = "SPLIT_CONSENSUS" # leafnode has equal number of positives and negatives
NO_DATA = "NO_DATA" # leafnode has no data to conduct split

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
        


def getInfoGain(data, meta, feature):
    """
        Function to compute information gain for a given numeric feature.
        Returns a numeric split and info gain for numeric features
        Returns only info gain for nominal features
    """

    feature_map = getUniqueFeatures(data, meta)

    # Step 1
    # For the selected feature, find the number of labels that have
    # positive and the number that have negative.
    # Using this compute the entropy for the feature itself

    parentEntropy = getEntropyParent(data, feature)

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

    if meta.types()[meta.names().index(feature)] == 'numeric':
        numericSplit, numericEntropy = getConditionalEntropy_Numeric(data, meta, feature, feature_map)
        infogain = parentEntropy - numericEntropy
        return numericSplit, infogain
    else:
        nominalEntropy = getConditionalEntropy_Nominal(data, meta, feature, feature_map)
        infogain = parentEntropy - nominalEntropy
        return None, infogain

def all_same(items):
    return all(x == items[0] for x in items)

def getConsensusClass(data, indices):
    
    a = collections.Counter(data['class'][indices])
    if all_same(list(a.values)) and len(list(a.values)) > 1:
        consensusClass = SPLIT_CONSENSUS
    else:
        consensusClass = a.most_common(1)[0][0]
    
    return consensusClass


def isLeafNode(data, meta, indices, m,  node):
    """
        tells us whether a given dataset is a
        leafnode or another feature branch
        Returns the modified leaf node if 
        the data truly represents a leaf node.
        Else, returns None
    """

    # step 1: check if all the instances belong to the same
    # class
    classes = set(data['class'][indices])
    if len(classes) == 1:
        node.setValue(classes.pop())
        node.isLeaf = True
        return node
    
    # step 2: check if the number of data points is less
    # than m
    if len(indices) < m:
        consensusClass = getConsensusClass(data, indices)
        node.setValue(consensusClass)
        node.isLeaf = True
        return node
    
    # step 3: iterate through all the feature
    # and find out if any of the features
    # have info gain
    infogains = []
    for feature in meta.names():
        split, infogain = getInfoGain(data[indices], meta, feature) 
        infogains.append(infogain)
    
    # are all of the values 0 or negative?
    # if so, make the leaf node here  based on consensus
    if all(np.array(infogains) <= 0):
        consensusClass = getConsensusClass(data, indices)
        node.setValue(consensusClass)
        node.isLeaf = True
        return node
    
    # Step 4: are there any more candidate splits left?
    # for this, we need to essentially check if all of the 
    # data values for a all features is only a single value
    # As this would indicate that there can be no more splits
    # made on a data set
    feature_diversity = []
    for feature in meta.names():
        feature_diversity.append(len(set(data[feature][indices])))
    
    if (all(np.array(feature_diversity) == 1)):
        consensusClass = getConsensusClass(data, indices)
        node.setValue(consensusClass)
        node.isLeaf = True
        return node
    
    return node
    


def createNode(data, meta, indices, m, featureTrack, feature_map):
    """
        This function recursively builds the tree for a given dataset,
        Returns a DTNode object.

    """
    
    print("--Creating Node")
    node = DTNode() # create the tree node first

    # check if any training data will be used for this node
    if len(indices) == 0:
        node.setValue(NO_DATA)
        node.isLeaf = True
        return node


    node = isLeafNode(data, meta, indices, m, node)
    if node.isLeafNode():
        # If it comes in here, then this means
        # that the node is a leaf node 
        return node
    else:
        # if it comes in here, that means 
        # that this node is not a leaf node and 
        # must be dealt with accordingly

        bestFeature = None
        bestInfoGain = 0
        bestSplit_Numeric = None # best split in case of numeric feature

        # iterate through each feature
        for feature,ftype in zip(meta.names(), meta.types()):
            split, infogain = getInfoGain(data, meta, feature)
            if infogain > bestInfoGain:
                bestInfoGain = infogain
                bestFeature = feature
                bestSplit_Numeric = split
        
        
        node.setFeature(bestFeature) # set the current node contain the bestFeature

        node.setNumericSplit(bestSplit_Numeric) # If it is a nominal value, then the
                                                # the split value would be none, and 
                                                # would accordingly be set in the DTNode 
                                                # object held by node

        # once we have chosen the best feature for this node.
        # check it off the featureTrack mapping if it
        # is a nominal feature. 
        if meta.types()[meta.names().index(bestFeature)] == 'nominal':
            
            # ensure that the tracker is switched to True
            featureTrack[bestFeature] = True

            # obtain all the children for this feature
            branches = feature_map[bestFeature]

            # for each branch value in the feature,
            # divide up the dataset in such a manner 
            # all of the values under that feature
            # correspond to that branch

            for branch in branches:
                sub_indices = []
                for index, value in enumerate(data[bestFeature]):
                    if branch == value:
                        sub_indices.append(index)


                # RECURSIVELY CALL THE CREATENODE FUNCTION
                # IN ORDER TO GET THE NEXT CHILD FOR 
                # THIS BRANCH
                childNode = createNode(data, meta, sub_indices, m, 
                                        featureTrack, feature_map)

                # check if the child node is a leaf and if
                # so does it have a split consensus on
                # the data. 
                # In this case, find out what the most 
                # occuring class label for the data
                # passed to this node consists of.
                # Using this information, compute
                # which label we must assign to this case. 
                if childNode.isLeafNode() and childNode.getValue() == SPLIT_CONSENSUS:
                    childNode.setValue(getConsensusClass(data, indices))
                elif childNode.isLeafNode() and childNode.getValue() == NO_DATA:
                    childNode.setValue(getConsensusClass(data, indices))

                node.setChild(bestFeature,childNode)


            

        

    

def train(data, meta, m):
    """
        Simple function to train the data 
    """
    # Once we are in this function, we must begin to work on tree building part.
    # So, this function will begin to execute the a tree building set of functions
    # here on out. 

    # Step 1
    # call a function node to create the first node of the tree


    indices = list(range(0,200)) 
    feature_map = getUniqueFeatures(data, meta)
    
    # we need a set up that allows us to track nominal features 
    # down lower in the branch in order to be able to ensure that
    # a nominal feature is not used again down the tree in subtrees
    # We will initialize each value in this nominal feature map to 
    # false. As we go down the tree and use a certain 
    # nominal feture, the mapping for this feature will be 
    # set to True, so that the feature will not be evaluated
    # for infogain. This only applies to nominal features 
    # and not numerical features because it is possible
    # to find a more optimal candidate split for numerical
    # features down the tree. 

    print("||Reached Train")

    featureTrack = dict()
    for feature, ftype in zip(meta.names(), meta.types()):
        if ftype == 'nominal':
            featureTrack[feature] = False

    print("||Creating Tree")
    rootnode = createNode(data, meta, indices, m, featureTrack, feature_map)
    return rootnode
