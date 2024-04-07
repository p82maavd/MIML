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
        """
        Get attributes values of the bag

        Returns
        -------
        attributes data: numpy array
            Values of the attributes of the bag

        """
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
        """
        Get features values of the bag

        Returns
        -------
        features data: numpy array
            Values of the features of the bag

        """
        return self.data[0:, 0:self.get_number_features()]

    def get_number_features(self):
        """
        Get numbers of features of the bag

        Returns
        ----------
         numbers of features: int
            Numbers of features of the bag
        """
        if self.dataset is not None:
            return self.dataset.get_number_features()
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
            return self.dataset.get_labels_name()
        else:
            raise Exception("The bag isn't in any dataset, so there is no label info")

    def get_labels(self):
        """
        Get labels values of the bag

        Returns
        -------
        labels data : numpy array
            Values of the labels of the bag

        """
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
        print("Self data: ", self.data)
        if instance.get_number_attributes() == self.get_number_attributes():
            self.data = np.vstack((self.data, instance.data))
        else:
            raise Exception("The number of attributes of the bag and the instance to be added are different.")

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
        """
        Get value of an attribute of the bag

        Parameters
        ----------
        instance : int
            Index of the instance in the bag

        attribute : int/String
            Index/Name of the attribute

        Returns
        -------
        value : float
            Value of the attribute
        """
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

        values: numpy array
            Values for the new attribute. If not provided, new values would be zero
        """
        if self.dataset is None:
            if position is None:
                position = len(self.data)
            if values is None:
                values = np.array([0]*self.get_number_instances())
            elif len(values) != self.get_number_instances():
                raise Exception("Incorrect number of values for the new attribute. Should be the same as number of "
                                "instances of the bag")
            self.data = np.insert(self.data, position, values, axis=1)
        else:
            raise Exception("Can't add an attribute to a bag assigned to a dataset")

    def delete_attribute(self, position):
        """
        Delete attribute of the bag

        Parameters
        ----------
        position : int
            Position of the attribute in the bag
        """
        if self.dataset is None:
            self.data = np.delete(self.data, position, axis=1)
        else:
            raise Exception("Can't delete an attribute of a bag assigned to a dataset")

    def set_dataset(self, dataset):
        """
        Set dataset which contains the bag

        Parameters
        ----------
        dataset : MIMLDataset
            Dataset for the bag
        """
        # TODO: Ver como gestionar lo de la info de los atributos que este siempre actualizado
        self.dataset = dataset

    def show_bag(self):
        """
        Show bag info in table format
        """
        # TODO: Check
        if self.dataset is None:
            table = [[self.key]+[""]*self.get_number_attributes()]
        else:
            table = [[self.key] + self.get_features_name() + self.get_labels_name()]
        count = 0
        for i in range(self.get_number_instances()):
            table.append([count] + list(self.get_instance(i).get_attributes()))
            count += 1
        print(tabulate(table, headers='firstrow', tablefmt="grid", numalign="center"))
