class MIMLDataset:
    def __init__(self) -> None:
        self.name = "undefined"
        self.attributes = []
        self.data = dict()
        self.numberlabels=0
    
    def setName(self, name):
        self.name = name
    def getName(self):
        return self.name
    
    def setAttributes(self, attributes):
        self.attributes=attributes
    def getAttributes(self):
        return self.attributes
    
    def setNumberLabels(self, labels):
        self.numberlabels=labels
    def getNumberLabels(self):
        return self.numberlabels
    
    def addBag(self,key,values,labels):
        self.data[key]=(values,labels)
    
    def showDataset(self):
        #TODO: Improve this to table style
        for keys,values in self.data.items():
            print(keys)
            print(values)