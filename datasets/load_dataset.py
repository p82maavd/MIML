import numpy as np
import pandas as pd

from data.miml_dataset import MIMLDataset


def load_dataset(file):
    """
    Function to load a dataset

    Parameters
    ----------
    file : string
        Path of the dataset file
    """
    if file[-4:] == ".csv":
        return load_dataset_csv(file)
    elif file[-5:] == ".arff":
        return load_dataset_arff(file)
    else:
        print("Error")
        # TODO: Control de errores


def load_dataset_csv(file):
    """
    Function to load a dataset in csv format

    Parameters
    ----------
    file : string
        Path of the dataset file
    """
    dataset = pd.read_csv(file, header=0)
    print(dataset.describe())
    # TODO: Hay que ver como diferenciar los atributos de las labels
    # TODO: Si no se puede implementar la funcionalidad de pandas "[]"
    # TODO: y poner atributos y labels como parametros opcionales quizas
    pass


def load_dataset_arff(file):
    """
    Function to load a dataset in arff format

    Parameters
    ----------
    file : string
        Path of the dataset file
    """
    dataset = MIMLDataset()
    arff_file = open(file)
    attribs_name = []
    labels_name = []
    flag = 0
    for line in arff_file:

        # Comprobamos que la cadena no contenga espacios en blanco a la izquierda ni que sea vacía
        line = line.lstrip()

        if not line or line.startswith("%"):
            continue

        if line.startswith("@"):

            if line.startswith("@relation"):
                dataset.set_name(line[line.find(" ") + 1:])
            elif line.startswith("@attribute bag relational"):
                flag = 1
            elif line.startswith("@end bag"):
                flag = 2
            elif flag == 1:
                attribs_name.append(line.split(" ")[1])
            elif flag == 2:
                labels_name.append(line.split(" ")[1])

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

            values_list = []
            for v in values:
                values_list.append(np.array([float(i) for i in v.split(',')]))

            dataset.add_bag(key, np.array(values_list), np.array([int(i) for i in labels.split(',')]))

    dataset.set_attributes(attribs_name)
    dataset.set_labels(labels_name)
    return dataset
