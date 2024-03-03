import numpy as np
from tabulate import tabulate

from data.bag import Bag
from data.instance import Instance


class MIMLDataset:
    """"
    Class to manage MIML data obtained from datasets
    """

    def __init__(self) -> None:

        # TODO: Si dataset leido en csv, el nombre poner el del archivo
        self.name = "undefined"
        self.attributes = dict()
        self.data = dict()

    def set_name(self, name):
        """
        Set function for dataset name

        Parameters
        ----------
        name : string
            Name of the dataset
        """
        self.name = name

    def get_name(self):
        """
         Get function for dataset name

        Returns
        ----------
        name : string
            Name of the dataset
        """
        return self.name

    def set_attributes(self, attributes):
        """
        Set function for dataset attributes name

        Parameters
        ----------
        attributes : List of string
            List of the attributes name of the dataset
        """
        if len(self.attributes) != 0:
            for attribute in self.attributes.keys():
                if self.attributes[attribute] == 0:
                    self.attributes.pop(attribute)
        for attribute in attributes:
            self.attributes[attribute] = 0

    def get_attributes(self):
        """
        Get function for dataset attributes name

        Returns
        ----------
        attributes : List of strings
            Attributes name of the dataset
        """
        attributes = []
        for attribute in self.attributes.keys():
            if self.attributes[attribute] == 0:
                attributes.append(attribute)
        return attributes

    def get_number_attributes(self):
        """
        Get numbers of attributes of the dataset

        Returns
        ----------
         numbers of attributes: int
            Numbers of attributes of the dataset
        """

        return len(self.get_attributes())

    def set_labels(self, labels):
        """
        Set function for dataset labels name

        Parameters
        ----------
        labels: List of string
            List of the labels name of the dataset
        """
        if len(self.attributes) != 0:
            for attribute in self.attributes.keys():
                if self.attributes[attribute] == 1:
                    self.attributes.pop(attribute)
        for label in labels:
            self.attributes[label] = 1

    def get_labels(self):
        """
        Get function for dataset labels name

        Returns
        ----------
        labels : List of strings
            Labels name of the dataset
        """
        labels = []
        for attribute in self.attributes.keys():
            if self.attributes[attribute] == 1:
                labels.append(attribute)
        return labels

    def get_number_labels(self):
        """
        Get numbers of labels of the dataset

        Returns
        ----------
        numbers of labels: int
            Numbers of labels of the dataset
        """

        return len(self.get_labels())

    def get_bag(self, key):
        """
        Get data of a bag of the dataset

        Returns
        ----------
        bag: ndarray
            Attributes and labels of a bag of the dataset
        """

        return self.data[key]

    def add_bag(self, bag: Bag):
        """
        Add a bag to the dataset

        Parameters
        ----------
        bag : Bag
            Bag to be added

        """
        bag.set_dataset(self)
        self.data[bag.key] = bag

    def get_instance(self, key, index):
        """

        Parameters
        ----------
        key : string
            Key of the bag
            
        index : int
            Index of the instance in the bag

        Returns
        -------
        instance : Instance
            Instance

        """
        # TODO: check
        return self.get_bag(key).get_instance(index)

    def add_instance(self, key, values):
        """

        Parameters
        ----------
        key : string
            Key of the bag
        values : ndarray
            Values of the instance to be inserted

        """
        instance = Instance(values)
        self.get_bag(key).add_instance(instance)

    def get_number_bags(self):
        """
        Get numbers of bags of the dataset

        Returns
        ----------
        numbers of bags: int
            Numbers of bags of the dataset
        """
        return len(self.data)

    def get_number_instances(self):

        return sum(self.data[bag].get_number_instances() for bag in self.data.keys())

    def set_attribute(self, key, index, attribute, value):
        """
            Update value from attributes

            Parameters
            ----------
            key : string
                Bag key of the dataset

            attribute: int
                Attribute of the dataset

            value: float
                New value for the update
            """
        
        self.get_instance(key, index).set_attribute_by_index(attribute, value)


    def show_dataset(self, head=None, attributes=None, labels=None):
        """"
        Function to show information about the dataset
        """
        # TODO: Formatearlo para que se vea bonito
        # TODO: Hacer algo como head y tail de pandas, ponerlo como parametro quizas, tambien lista atributos y labels
        #  a mostrar opcionales
        print("Name: ", self.get_name())
        print("Attributes: ", self.get_attributes())
        print("Labels: ", self.get_labels())
        print("Bags:")
        count = 0
        for key in self.data:
            # print("\n")
            bag = self.get_bag(key)
            # print("Key: ", key)
            # print("Attributes: ", bag[0])
            # print("Labels: ", bag[1])
            bag.show_bag()
            count += 1
            if head is not None:
                if count >= head:
                    break
            # print(bag)

    def cardinality(self):
        """
        Computes the Cardinality as the average number of labels per pattern.

        Returns
        ----------
        cardinality : float
            Average number of labels per pattern
        """
        suma = 0
        for key in self.data:
            suma += sum(self.data[key][1])
        return suma / len(self.data)

    def density(self):
        """
        Computes the density as the cardinality / numLabels.

        Returns
        ----------
        density : float
            Cardinality divived by number of labels
        """
        return self.cardinality() / self.get_number_labels()

    def distinct(self):
        """
        Computes the numbers of labels combinations used in the dataset respect all the possible ones
        
        Returns
        -------
        distinct : float
            Numbers of labels combinations used in the dataset divided by all possible combinations
        """
        options = set()
        for key in self.data:
            options.add(tuple(self.data[key][1]))
        return len(options) / (2 ** self.get_number_labels())

    def get_statistics(self):
        n_instances = 0
        max_instances = 0
        # TODO: check
        min_instances = float("inf")
        distribution = dict()
        for key in self.data:
            instances_bag = len(self.get_bag(key)[0])
            n_instances += instances_bag
            if instances_bag in distribution:
                distribution[instances_bag] += 1
            else:
                distribution[instances_bag] = 1
            if instances_bag < min_instances:
                min_instances = instances_bag
            elif instances_bag > max_instances:
                max_instances = instances_bag
        return n_instances, min_instances, max_instances, distribution

    def describe(self):
        """
        Print statistics about the dataset
        """

        # TODO: Ponerlo bonito con tabulate

        print("-----MULTILABEL-----")
        print("Cardinalidad: ", self.cardinality())
        print("Densidad: ", self.density())
        print("Distinct: ", self.distinct())
        print("")
        # TODO: Testearlo
        n_instances, min_instances, max_instances, distribution = self.get_statistics()
        print("-----MULTIINSTANCE-----")
        print("Nº of bags: ", self.get_number_bags())
        print("Total instances: ", n_instances)
        print("Average Instances per bag: ", n_instances / self.get_number_bags())
        print("Min Instances per bag: ", min_instances)
        print("Max Instances per bag: ", max_instances)
        print("Attributes per bag: ", self.get_number_attributes())
        # TODO: Implementarlo
        # sb.append("\nDistribution of bags <nBags, nInstances>:");
