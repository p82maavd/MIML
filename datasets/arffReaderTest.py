import numpy as np

#Clase para gestionar archivos .arff para problemas multi-etiqueta multi-instancia
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
    
    for line in arff_file:
            
            #Comprobamos que la cadena no contenga espacios en blanco a la izquierda ni que sea vacía
            line=line.lstrip()
            if line=="":
                continue
            
            if not (line.startswith("@")):
                if not (line.startswith("%")):
                    
                    #Eliminanos el salto de línea del final de la cadena
                    line=line.strip("\n")

                    #Asumimos que el primer elemento de cada instancia es el identificador de la bolsa
                    key = line[0:line.find(",")]
                    #print("Key: ", key)

                    #Empiezan los datos de la bolsa cuando encontremos la primera '"' y terminan con la segunda '"'
                    line=line[line.find("\"")+1:]
                    values = line[:line.find("\"",2)]
                    #Separamos los valores por instancias de la bolsa
                    values=values.split("\\n")
                    #print("Values ", values)

                    #El resto de la cadena se trata de las etiquetas
                    labels=line[line.find("\"",2)+2:]
                    #print("Labels: ", labels)
                    
                    valueslist=[]
                    for v in values:
                        valueslist.append(np.array([float(i) for i in v.split(',')]))
                    
                    arff.addBag(key,np.array(valueslist),np.array([int(i) for i in labels.split(',')]))
                            #TODO: añadir gestion atributos
                            #TODO: quizas separar en funciones para data y para atributos
                            #TODO: incluso diccionario aparte para stats
    
    return arff

toy=arffMIMLReader("toy.arff")
toy.showArff()

birds=arffMIMLReader("miml_birds.arff")
#birds.showArff()