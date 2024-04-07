import numpy as np
from tabulate import tabulate


class Instance:
    """
    Class to manage MIML Instance data representation
    """

    def __init__(self, values=None, bag=None):
        """
        Constructor of the class Instance
        """
        self.bag = bag
        self.data = np.array(values)

    def get_attributes_name(self):
        """
        Get attributes name

        Returns
        ----------
        attributes : List of string
            Attributes name of the instance
        """
        if self.bag is not None:
            return self.bag.get_attributes_name()
        else:
            # TODO: Control de errores
            return 0

    def get_attributes(self):
        """
        Get data attributes of the instance

        Returns
        ----------
        attributes data: numpy array
            Values of the attributes of the instance
        """
        return self.data

    def get_number_attributes(self):
        """
        Get numbers of attributes of the instance

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
            features name of the instance
        """
        if self.bag is not None:
            return self.bag.get_features_name()
        else:
            # TODO: Control de errores
            return 0

    def get_features(self):
        return self.data[0:self.get_number_features()]

    def get_number_features(self):
        """
        Get numbers of features of the instance

        Returns
        ----------
         numbers of features: int
            Numbers of features of the bag
        """
        if self.bag is not None:
            return self.bag.get_number_features()
        else:
            # TODO: Control de errores
            return 0

    def get_labels_name(self):
        """
        Get labels name

        Returns
        ----------
        labels : List of string
            Labels name of the instance
        """
        if self.bag is not None:
            return self.bag.get_labels()
        else:
            # TODO: Control de errores
            return 0

    def get_labels(self):
        # TODO: test
        return self.data[-self.get_number_labels():]

    def get_number_labels(self):
        """
        Get numbers of labels of the instance

        Returns
        ----------
        numbers of labels : int
            Numbers of labels of the instance
        """
        if self.bag is not None:
            return self.bag.get_number_labels()
        else:
            # TODO: Control de errores
            return 0

    def get_attribute(self,  attribute):
        """
        Get value of an attribute of the instance

        Parameters
        ----------
        attribute : int/String
            Index/Name of the attribute

        Returns
        -------
        value : float
            Value of the attribute
        """
        if isinstance(attribute, int):
            return self.data.item(attribute)
        elif isinstance(attribute, str):
            index = list(self.get_attributes()).index(attribute)
            return self.data.item(index)

    def set_attribute(self, attribute, value):
        """
        Update value of a attribute of the instance

        Parameters
        ----------
        attribute : int/String
            Index/Name of the attribute

        value : float
            New value for the attribute
        """
        if isinstance(attribute, int):
            self.data[attribute] = value
        elif isinstance(attribute, str):
            index = list(self.get_attributes()).index(attribute)
            self.data[index] = value

    def add_attribute(self, value=0, position=None):
        """
        Add an attribute to the instance

        Parameters
        ----------
        value : float
            Value for the attribute

        position: int
            Position for the attribute
        """
        if self.bag is None:
            if position is None:
                position = len(self.data)
            self.data = np.insert(self.data, position, value)
        else:
            # TODO: Control de errores
            pass

    def delete_attribute(self, position):
        """
        Delete an attribute of the instance

        Parameters
        ----------
        position: int
            Position of the attribute
        """
        # TODO: Check
        if self.bag is None:
            self.data = np.delete(self.data, position)
        else:
            # TODO: Control de errores
            pass

    def set_bag(self, bag):
        """
        Set the bag of the instance

        Parameters
        ----------
        bag : Bag
            Bag of the instance
        """
        self.bag = bag

    def show_instance(self):
        """
        Show instance info in table format
        """
        # TODO: Check. Hacer que muestre el nombre de los atributos.

        table = list(self.get_attributes())

        # table = [['col 1', 'col 2', 'col 3', 'col 4'], [1, 2222, 30, 500], [4, 55, 6777, 1]]
        # print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        # print(tabulate([key], tablefmt="grid"))
        print(tabulate(table, tablefmt="grid", numalign="center"))
