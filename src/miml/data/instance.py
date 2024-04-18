import numpy as np
from tabulate import tabulate


class Instance:
    """
    Class to manage MIML Instance data representation
    """

    def __init__(self, values=None, bag=None) -> None:
        """
        Constructor of the class Instance
        """
        self.bag = bag
        self.data = np.array(values)

    def get_attributes_name(self) -> list[str]:
        """
        Get attributes name

        Returns
        ----------
        attributes : List of string
            Attributes name of the instance
        """
        if self.bag is not None and self.bag.dataset is not None:
            return self.bag.dataset.get_attributes_name()
        else:
            raise Exception("The instance isn't in any dataset, so there is no attributes info")

    def get_attributes(self) -> np.ndarray:
        """
        Get data attributes of the instance

        Returns
        ----------
        attributes data: numpy array
            Values of the attributes of the instance
        """
        return self.data

    def get_number_attributes(self) -> int:
        """
        Get numbers of attributes of the instance

        Returns
        ----------
         numbers of attributes: int
            Numbers of attributes of the bag
        """
        return len(self.get_attributes())

    def get_features_name(self) -> list[str]:
        """
        Get features name

        Returns
        ----------
        features : List of string
            features name of the instance
        """
        if self.bag is not None and self.bag.dataset is not None:
            return self.bag.dataset.get_features_name()
        else:
            raise Exception("The instance isn't in any dataset, so there is no features info")

    def get_features(self) -> np.ndarray:
        """
        Get features values of the instance

        Returns
        -------
        features data: numpy array
            Values of the features of the instance

        """
        return self.data[0:self.get_number_features()]

    def get_number_features(self) -> int:
        """
        Get numbers of features of the instance

        Returns
        ----------
         numbers of features: int
            Numbers of features of the bag
        """
        if self.bag is not None and self.bag.dataset is not None:
            return self.bag.dataset.get_number_features()
        else:
            raise Exception("The instance isn't in any dataset, so there is no features info")

    def get_labels_name(self) -> list[str]:
        """
        Get labels name

        Returns
        ----------
        labels : List of string
            Labels name of the instance
        """
        if self.bag is not None and self.bag.dataset is not None:
            return self.bag.dataset.get_labels_name()
        else:
            raise Exception("The instance isn't in any dataset, so there is no labels info")

    def get_labels(self) -> np.ndarray:
        """
        Get labels values of the instance

        Returns
        -------
        labels data : numpy array
            Values of the labels of the instance

                """
        return self.data[-self.get_number_labels():]

    def get_number_labels(self) -> int:
        """
        Get numbers of labels of the instance

        Returns
        ----------
        numbers of labels : int
            Numbers of labels of the instance
        """
        if self.bag is not None and self.bag.dataset is not None:
            return self.bag.dataset.get_number_labels()
        else:
            raise Exception("The instance isn't in any dataset, so there is no labels info")

    def get_attribute(self,  attribute) -> float:
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

    def set_attribute(self, attribute, value) -> None:
        """
        Update value of an attribute of the instance

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

    def add_attribute(self, value=0, position=None) -> None:
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
            raise Exception("Can't add an attribute to an instance assigned to a bag")

    def delete_attribute(self, position) -> None:
        """
        Delete an attribute of the instance

        Parameters
        ----------
        position: int
            Position of the attribute
        """
        if self.bag is None:
            self.data = np.delete(self.data, position)
        else:
            raise Exception("Can't delete an attribute of a bag assigned to a bag")

    def set_bag(self, bag) -> None:
        """
        Set the bag of the instance

        Parameters
        ----------
        bag : Bag
            Bag of the instance
        """
        self.bag = bag

    def show_instance(self) -> None:
        """
        Show instance info in table format
        """
        table = []
        if self.bag is not None and self.bag.dataset is not None:
            table = [self.get_features_name() + self.get_labels_name()]
        table.append(list(self.get_attributes()))
        print(tabulate(table, tablefmt="grid", numalign="center"))
