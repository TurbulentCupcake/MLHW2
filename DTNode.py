

class DTNode(object):
    """
        Node for each tree.
    """

    def __init__(self):
        self.feature = None
        self.children = dict()
        self.isLeaf = False
        self.value = None
        self.numeric_split = None
        self.featuretype = None
        self.leftchild = None
        self.rightchild = None
    

    def getFeature(self):
        return self.feature
    
    def setFeature(self, feature):
        self.feature = feature

    def isLeafNode(self):
        return self.isLeaf

    def setValue(self, value):
        self.value = value
    
    def getValue(self):
        return self.value

    def setChild(self, value, child):
        self.children[value] = child
    
    def getChild(self, value):
        return self.children[value]
    
    def getChildren(self):
        return self.children
    
    def setNumericSplit(self, numeric_split):
        self.numeric_split = numeric_split
    
    def getNumeicSplit(self):
        return self.numeric_split
    
    def setFeatureType(self, featureType):
        self.featuretype = featureType
    
    def getFeatureType(self):
        return self.featuretype

    def setRightChild(self, right_child):
        self.rightchild = right_child
    
    def setLeftChild(self, left_child):
        self.leftchild = left_child
    
    def getRightChild(self):
        return self.rightchild
    
    def getLeftChild(self):
        return self.leftchild
    



