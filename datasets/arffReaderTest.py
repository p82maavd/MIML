class MIMLArff:
    def __init__(self) -> None:
        self.name = "undefined"
        self.attributes = []
        self.data = dict()
    
    def setName(self, name):
        self.name = name
    def getName(self):
        return self.name
    
    def setAttributes(self, attributes):
        self.attributes=attributes
    def getAttributes(self):
        return self.attributes
    
    def addBag(self,key,values,labels):
        self.data[key]=(values,labels)
    
    def showArff(self):
        #TODO: Improve this to table style
        for keys,values in self.data.items():
            print(keys)
            print(values)


def arffMIMLReader(file):

    arff = MIMLArff()
    arff_file = open(file)
    count=0
    for line in arff_file:
            count+=1
            line=line.lstrip()
            if line=="":
                continue
            
            if not (line.startswith("@")):
                if not (line.startswith("%")):
                    line=line.strip("\n")
                    #line=line.split(',')
                    #print(line)
                    key = line[0:line.find(",")]
                    line=line[line.find("\"")+1:]
                    labels=line[line.find("\"",2)+2:]
                    values = line[:line.find("\"",2)].split("\\n")
                    
                    #print("Key: ", key)
                    #print("Values ", values)
                    valueslist=[]
                    for v in values:
                        valueslist.append(tuple([float(i) for i in v.split(',')]))
                    #print("Labels: ", labels)
                    
                    arff.addBag(key,tuple(valueslist),tuple([int(i) for i in labels.split(',')]))
                            #Cuando se llegue a data, se añade el id del bag a un diccionario y los values de cada clave 
                            #sera una tupla de 2 elementos, una tupla con tuplas de los distintos values y otra tupla con las etiquetas
                            #key: (((3,5,6,7),(1,4,5,7)),(1,0,1))
                            #TODO: incluso diccionario aparte para stats
    
    return arff

toy=arffMIMLReader("toy.arff")
toy.showArff()

birds=arffMIMLReader("miml_birds.arff")
#birds.showArff()