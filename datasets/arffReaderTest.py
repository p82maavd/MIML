from ast import pattern
from turtle import shape
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss

#Clase para gestionar archivos .arff para problemas multi-etiqueta multi-instancia
class MIMLArff:
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
    
    def showArff(self):
        #TODO: Improve this to table style
        for keys,values in self.data.items():
            print(keys)
            print(values)



def arffMIMLReader(file):

    arff = MIMLArff()
    arff_file = open(file)
    attrib=[]
    flag=0
    for line in arff_file:
            
            #Comprobamos que la cadena no contenga espacios en blanco a la izquierda ni que sea vacía
            line=line.lstrip()
            if line=="":
                continue
            
            
            if line.startswith("@"):
                if not (line.startswith("%")):
                    if line.startswith("@relation"):
                        arff.setName(line[line.find(" ")+1:])
                    elif line.startswith("@attribute bag relational"):
                        flag = 1
                    elif line.startswith("@end bag"):
                        flag = 0
                    elif flag==1:
                        if line.startswith("@attribute"):
                            
                            attrib.append(line[line.find(" ")+1:line.find(" ",line.find(" ")+1)])
            
                
            else:        
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
                arff.setNumberLabels(len(labels.split(",")))
                #print("Labels: ", labels)
                        
                valueslist=[]
                for v in values:
                    valueslist.append(np.array([float(i) for i in v.split(',')]))
                        
                arff.addBag(key,np.array(valueslist),np.array([int(i) for i in labels.split(',')]))
                    #TODO: añadir gestion atributos
                    #TODO: quizas separar en funciones para data y para atributos
                    #TODO: incluso diccionario aparte para stats

    arff.setAttributes(attrib)    
    return arff

def arffMIMLReaderComillaSimple(file):

    arff = MIMLArff()
    arff_file = open(file)
    attrib=[]
    flag=0
    for line in arff_file:
            
            #Comprobamos que la cadena no contenga espacios en blanco a la izquierda ni que sea vacía
            line=line.lstrip()
            if line=="":
                continue
            
            
            if line.startswith("@"):
                if not (line.startswith("%")):
                    if line.startswith("@relation"):
                        arff.setName(line[line.find(" ")+1:])
                    elif line.startswith("@attribute bag relational"):
                        flag = 1
                    elif line.startswith("@end bag"):
                        flag = 0
                    elif flag==1:
                        if line.startswith("@attribute"):
                            
                            attrib.append(line[line.find(" ")+1:line.find(" ",line.find(" ")+1)])
            
                
            else:        
                #Eliminanos el salto de línea del final de la cadena
                line=line.strip("\n")

                #Asumimos que el primer elemento de cada instancia es el identificador de la bolsa
                key = line[0:line.find(",")]
                #print("Key: ", key)

                #Empiezan los datos de la bolsa cuando encontremos la primera '"' y terminan con la segunda '"'
                line=line[line.find("\'")+1:]
                values = line[:line.find("\'",2)]
                #Separamos los valores por instancias de la bolsa
                values=values.split("\\n")
                #print("Values ", values)

                #El resto de la cadena se trata de las etiquetas
                labels=line[line.find("\'",2)+2:]
                arff.setNumberLabels(len(labels.split(",")))
                #print("Labels: ", labels)
                        
                valueslist=[]
                for v in values:
                    valueslist.append(np.array([float(i) for i in v.split(',')]))
                        
                arff.addBag(key,np.array(valueslist),np.array([int(i) for i in labels.split(',')]))
                    #TODO: añadir gestion atributos
                    #TODO: quizas separar en funciones para data y para atributos
                    #TODO: incluso diccionario aparte para stats

    arff.setAttributes(attrib)    
    return arff

def convertToMultiLabel(arff):
    #TODO: Media aritmetica, geometrica, min-max    
    x = np.empty(shape=(len(arff.data.keys()),len(arff.getAttributes())))
    print("Number of labels: ",arff.getNumberLabels())
    y = np.empty(shape=(len(arff.data.keys()),arff.getNumberLabels()))
    count=0
    for keys,pattern in arff.data.items():
            newinstance=np.zeros(pattern[0][0].shape[0])
            for instance in pattern[0]:
                newinstance+=instance
            newinstance/=pattern[0].shape[0]
            #print(newinstance)
            x[count]=newinstance
            print(pattern[1])
            y[count]=pattern[1]
            count+=1
    print("X Data: ",x)
    print("Y Data: ",y)

    return x,y



#toy=arffMIMLReader("toy.arff")
#toy.showArff()
#print(toy.getName())
#print(toy.getAttributes())
#convertToMultiLabel(toy)


birdstrain=arffMIMLReaderComillaSimple("miml_birds_random_80train.arff")
X_train,y_train = convertToMultiLabel(birdstrain)

birdstest=arffMIMLReaderComillaSimple("miml_birds_random_20test.arff")
X_test, y_test= convertToMultiLabel(birdstest)


classifier = MultiOutputClassifier(RandomForestClassifier(random_state=27))
classifier.fit(X_train, y_train)

# Predicciones
y_pred = classifier.predict(X_test)

# Evaluación del modelo
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

#print("Y TEST:",y_test)
#print("Y Pred:",y_pred)

print('Hamming Loss: ', round(hamming_loss(y_test, y_pred),2))





