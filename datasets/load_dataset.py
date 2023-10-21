import numpy as np
import pandas as pd

from data.miml_dataset import *


def load_dataset(file):
    if file[-4:] == ".csv":
        return load_dataset_csv(file)
    elif file[-5:] == ".arff":
        return load_dataset_arff(file)
    else:
        print("Error")
        # TODO: Control de errores


def load_dataset_csv(file):
    dataset = pd.read_csv(file, header=0)
    print(dataset.describe())
    # TODO
    pass


def load_dataset_arff(file):
    dataset = MIMLDataset()
    arff_file = open(file)
    attrib = []
    labels_name = []
    flag = 0
    for line in arff_file:

        # Comprobamos que la cadena no contenga espacios en blanco a la izquierda ni que sea vacía
        line = line.lstrip()
        if line == "":
            continue

        if line.startswith("@"):
            if not (line.startswith("%")):
                if line.startswith("@relation"):
                    dataset.set_name(line[line.find(" ") + 1:])
                elif line.startswith("@attribute bag relational"):
                    flag = 1
                elif line.startswith("@end bag"):
                    flag = 2
                elif flag == 1:
                    if line.startswith("@attribute"):
                        attrib.append(line[line.find(" ") + 1:line.find(" ", line.find(" ") + 1)])
                elif flag == 2:
                    if line.startswith("@attribute"):
                        labels_name.append(line[line.find(" ") + 1:line.find(" ", line.find(" ") + 1)])

        else:
            # Eliminanos el salto de línea del final de la cadena
            line = line.strip("\n")

            # Asumimos que el primer elemento de cada instancia es el identificador de la bolsa
            key = line[0:line.find(",")]
            # print("Key: ", key)

            # Empiezan los datos de la bolsa cuando encontremos la primera '"' y terminan con la segunda '"'
            line = line[line.find("\"") + 1:]
            values = line[:line.find("\"", 2)]
            # Separamos los valores por instancias de la bolsa
            values = values.split("\\n")
            # print("Values ", values)

            # El resto de la cadena se trata de las etiquetas
            labels = line[line.find("\"", 2) + 2:]
            # print("Labels: ", labels)

            valueslist = []
            for v in values:
                valueslist.append(np.array([float(i) for i in v.split(',')]))

            dataset.add_bag(key, np.array(valueslist), np.array([int(i) for i in labels.split(',')]))
            # TODO: añadir gestion atributos
            # TODO: quizas separar en funciones para data y para atributos
            # TODO: incluso diccionario aparte para stats

    dataset.set_attributes(attrib)
    dataset.set_labels(labels_name)
    return dataset

# load_dataset("miml_birds.csv")
