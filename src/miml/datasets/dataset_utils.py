import numpy as np
import os

import pkg_resources

from src.miml.data.bag import Bag
from src.miml.data.instance import Instance
from src.miml.data.miml_dataset import MIMLDataset


def load_dataset(file: str, delimiter: str = "\"") -> MIMLDataset:
    """
    Function to load a dataset

    Parameters
    ----------
    file : str
        Path of the dataset file
    delimiter : str
        Character to separate instances in bag of the arff files

    Returns
    ----------
    dataset : MIMLDataset
        Dataset loaded
    """
    if file[-4:] == ".csv":
        return load_dataset_csv(file)
    elif file[-5:] == ".arff":
        return load_dataset_arff(file, delimiter)
    else:
        print("Error")
        # TODO: Control de errores


def load_dataset_csv(file: str, header=0):
    """
    Function to load a dataset in csv format

    Parameters
    ----------
    file : str
        Path of the dataset file

    Returns
    -------
    dataset : MIMLDataset
        Dataset loaded
    """

    dataset = MIMLDataset()
    csv_file = open(file)
    dataset.set_name(file.split("/")[-1])
    file_name = os.path.basename(file)
    dataset.set_name(os.path.splitext(file_name)[0])

    # TODO: Make it possible to pass by parameter
    num_labels = int(csv_file.readline().replace("\n", ""))

    header_line = csv_file.readline().replace("\n", "").split(",")
    features_name = header_line[1:-num_labels]
    dataset.set_features_name(features_name)
    labels_name = header_line[-num_labels:]
    dataset.set_labels_name(labels_name)

    for line in csv_file:

        data = line.split(",")

        key = data[0]

        values = [float(i) for i in data[1:-num_labels]]
        labels = [int(i) for i in data[-num_labels:]]

        instance = Instance(values + labels)

        if key not in dataset.data:
            bag = Bag(key)
            dataset.add_bag(bag)
        dataset.add_instance(key, instance)

    return dataset


def load_dataset_arff(file: str, delimiter: str = "\"") -> MIMLDataset:
    """
    Function to load a dataset in arff format

    Parameters
    ----------
    file : str
        Path of the dataset file
    delimiter : str
        Delimiter of instances in a bag in the arff file

    Returns
    -------
    dataset : MIMLDataset
        Dataset loaded
    """
    dataset = MIMLDataset()
    arff_file = open(file)
    features_name = []
    labels_name = []
    flag = 0
    for line in arff_file:

        # We check that the string does not contain spaces on the left and that it is not empty.
        line = line.lstrip()

        if not line or line.startswith("%") or line.startswith("@data"):
            continue

        if line.startswith("@"):

            if line.startswith("@relation"):
                dataset.set_name(line[line.find(" ") + 1:])
            elif line.startswith("@attribute bag relational"):
                flag = 1
            elif line.startswith("@end bag"):
                flag = 2
            elif flag == 1:
                features_name.append(line.split(" ")[1])
                dataset.set_features_name(features_name)
            elif flag == 2:
                labels_name.append(line.split(" ")[1])
                dataset.set_labels_name(labels_name)

        else:
            # Remove line break at the end of the string
            line = line.strip("\n")

            # We assume that the first element of each instance is the bag identifier
            key = line[0:line.find(",")]

            # Start the bag data when we find the first '"' and end with the second '"'.
            line = line[line.find(delimiter) + 1:]
            values = line[:line.find(delimiter, 2)]
            # Separamos los valores por instancias de la bolsa
            values = values.split("\\n")

            # The rest of the string consists of the following labels
            labels = line[line.find(delimiter, 2) + 2:]
            labels_values = [int(i) for i in labels.split(",")]

            for v in values:
                values_instance = [float(i) for i in v.split(',')]
                instance = Instance(values_instance + labels_values)
                if key not in dataset.data:
                    bag = Bag(key)
                    bag.add_instance(instance)
                    dataset.add_bag(bag)
                else:
                    dataset.add_instance(key, instance)

    return dataset


def load_toy():
    # TODO: Doc
    return load_dataset(pkg_resources.resource_filename('miml', 'datasets/toy.arff'), delimiter="'")


def load_birds():
    # TODO: Doc
    return load_dataset(pkg_resources.resource_filename('miml', 'datasets/miml_birds.arff'),
                        delimiter="'")


def load_birds_train():
    # TODO: Doc
    return load_dataset(pkg_resources.resource_filename('miml', 'datasets/miml_birds_random_80train.arff'),
                        delimiter="'")


def load_birds_test():
    # TODO: Doc
    return load_dataset(pkg_resources.resource_filename('miml', 'datasets/miml_birds_random_20test.arff'),
                        delimiter="'")


def arff_to_csv(file: str, delimiter: str = "'") -> None:
    """
    Convert MIML Arff to CSV.

    Parameters
    ----------
    file : str
        Filepath of the file to be converted

    delimiter : str, default = '
        Delimiter used in arff file for the start and end of the bag values ( ' or " )
    """

    arff = open(file)
    csv = open(file[:-5] + ".csv", "w")
    attrib = []
    flag = 0

    for line in arff:
        # Comprobamos que la cadena no contenga espacios en blanco a la izquierda ni que sea vacía
        line = line.lstrip()
        if line == "":
            continue

        if line.startswith("%") or line.startswith("@"):
            if line.startswith("@attribute") and not line.startswith("@attribute bag relational"):
                attrib.append(line.split()[1])

        else:
            if flag == 0:
                csv.write(','.join(attrib) + '\n')
                flag = 1

            # Eliminanos el salto de línea del final de la cadena
            line = line.strip("\n")

            # Asumimos que el primer elemento de cada instancia es el identificador de la bolsa
            key = line[0:line.find(",")]
            # print("Key: ", key_bag)

            # Empiezan los datos de la bolsa cuando encontremos la primera '"' y terminan con la segunda '"'
            line = line[line.find(delimiter) + 1:]
            values = line[:line.find(delimiter, line.find(delimiter, line.find(delimiter)))]
            # Separamos los valores por instancias de la bolsa
            values = values.split("\\n")
            # print("Values ", values)

            # El resto de la cadena se trata de las etiquetas
            labels = line[line.find(delimiter, line.find(delimiter, line.find(delimiter))) + 2:]
            # print("Labels: ", labels)

            for v in values:
                csv.write(key + "," + v + "," + labels + "\n")
                # TODO: intentar optimizar la funcion, aprovechar que la string del arff
                # es parecida, separar solo los values
                # TODO: control de errores

    arff.close()
    csv.close()

