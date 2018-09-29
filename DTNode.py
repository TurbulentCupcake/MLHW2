

class DTNode(object):
    """
        Node for each tree.
    """

    def __init__(self):
        self.feature = None
        self.children = dict()
        self.isLeaf = False
        self.value = None
    

    def getFeature(self):
        return self.feature
    
    def getChildren(self):
        return self.children
    
    def setFeature(self, feature):
        self.feature = feature
    
    def addChild(self, child):
        self.children.append(child)
    
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
    



