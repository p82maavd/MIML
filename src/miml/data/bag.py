import numpy as np
from data.instance import Instance
from tabulate import tabulate


class Bag:
    """
    Class to manage MIML Bag data representation
    """

    def __init__(self, instance, key):
        """
        Constructor of the class Bag
        """
        # TODO: Ver si quitar instance del constructor
        self.data = np.array(instance.data)
        self.key = key
        self.dataset = None

    def get_attributes_name(self):
        """
        Get attributes name

        Returns
        ----------
        attributes : List of string
            Attributes name of the bag
        """
        if self.dataset is not None:
            return self.dataset.get_attributes_name()
        else:
            raise Exception("The bag isn't in any dataset, so there is no attributes info")

    def get_attributes(self):
        return self.data

    def get_number_attributes(self):
        """
        Get numbers of attributes of the bag

        Returns
        ----------
         numbers of attributes: int
            Numbers of attributes of the bag
        """
        return len(self.get_attributes())

    def get_features_name(self):
        """
        Get features name

        Returns
        ----------
        features : List of string
            Features name of the bag
        """
        if self.dataset is not None:
            return self.dataset.get_features_name()
        else:
            raise Exception("The bag isn't in any dataset, so there is no features info")

    def get_features(self):
        return self.data[0:, 0:self.get_number_features()]

    def get_number_features(self):
        """
        Get numbers of attributes of the bag

        Returns
        ----------
         numbers of attributes: int
            Numbers of attributes of the bag
        """
        if self.dataset is not None:
            return self.dataset.get_features_name()
        else:
            raise Exception("The bag isn't in any dataset, so there is no features info")

    def get_labels_name(self):
        """
        Get labels name

        Returns
        ----------
        labels : List of string
            Labels name of the bag
        """
        if self.dataset is not None:
            return self.dataset.get_labels()
        else:
            raise Exception("The bag isn't in any dataset, so there is no label info")

    def get_labels(self):
        return self.data[0, -self.get_number_labels():]

    def get_number_labels(self):
        """
        Get numbers of labels of the bag

        Returns
        ----------
        numbers of labels: int
            Numbers of labels of the bag
        """
        if self.dataset is not None:
            return self.dataset.get_number_labels()
        else:
            raise Exception("The bag isn't in any dataset, so there is no label info")

    def get_instance(self, index):
        """
        Get an Instance of the Bag

        Parameters
        ----------

        index : int
            Index of the instance in the bag

        Returns
        -------
        instance : Instance
            Instance of Instance class
        """
        instance = Instance(self.data[index], self)
        return instance

    def get_number_instances(self):
        """
        Get numbers of instances of a bag

        Returns
        ----------
        numbers of instances: int
            Numbers of instances of a bag
        """
        return len(self.data)

    def add_instance(self, instance):
        """
        Add instance to the bag

        Parameters
        ----------
        instance : Instance
            Instance to be added
        """
        # TODO: Check same length
        self.data = np.vstack((self.data, instance.data))

    def delete_instance(self, index):
        """
        Delete a instance of the bag

        Parameters
        ----------
        index : int
            Index of the instance to be removed
        """
        self.data = np.delete(self.data, index, axis=0)

    def get_attribute(self, instance, attribute):

        if isinstance(attribute, int):
            return self.data[instance].item(attribute)
        elif isinstance(attribute, str):
            index = list(self.get_attributes()).index(attribute)
            return self.data[instance].item(index)

    def set_attribute(self, instance, attribute, value):
        """
        Update value from attributes

        Parameters
        ----------
        instance : string
            Index of instance to me update

        attribute: string
            Attribute name/index of the bag to be updated

        value: float
            New value for the update
        """
        if isinstance(attribute, int):
            self.data[instance][attribute] = value
        elif isinstance(attribute, str):
            index = list(self.get_attributes()).index(attribute)
            self.data[instance][index] = value

    def add_attribute(self, position, values=None):
        """
        Add attribute to the bag

        Parameters
        ----------
        position : int
            Index for the new attribute

        values: 1d numpy array
            Values for the new attribute. If not provided, new values would be zero
        """
        if self.dataset is None:
            if position is None:
                position = len(self.data)
            if values is None:
                values = np.array([0]*self.get_number_instances())
            else:
                # TODO: Check size len(values) == self.get_number_instances
                pass
            self.data = np.insert(self.data, position, values, axis=1)
        else:
            pass

    # TODO: Terminar funcion

    def delete_attribute(self, position):
        if self.dataset is None:
            self.data = np.delete(self.data, position, axis=1)
        else:
            pass

    def set_dataset(self, dataset):
        # TODO: Ver como gestionar lo de la info de los atributos que este siempre actualizado
        self.dataset = dataset

    def show_bag(self):
        # TODO: Check

        table = [[self.key] + self.get_attributes_name() + self.get_labels_name()]
        count = 0
        for instance in self.data:
            table.append([count] + list(instance))
            count += 1
        # table = [['col 1', 'col 2', 'col 3', 'col 4'], [1, 2222, 30, 500], [4, 55, 6777, 1]]
        # print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        # print(tabulate([key], tablefmt="grid"))
        print(tabulate(table, headers='firstrow', tablefmt="grid", numalign="center"))
