import numpy as np
from data.instance import Instance
from tabulate import tabulate


class Bag:

    def __init__(self, instance, key):
        # Crear numpy ndarray 3d con una instancia
        self.data = np.array(instance.data)
        self.key = key
        self.dataset = None

    def get_number_instances(self):
        """
        Get numbers of instances of a bag

        Parameters
        ----------
        key : string
            Key of the bag

        Returns
        ----------
        numbers of instances: int
            Numbers of instances of a bag

        """
        # TODO: Revisar
        return len(self.data)

    def get_instance(self, index):
        """

        Parameters
        ----------

        index : int
            Index of the instance in the bag

        Returns
        -------
        instance : tuple of ndarrays
            Tuple with attribute values and label of the instance

        """
        instance = Instance(self.data[index], self)
        return instance

    def get_attributes(self):
        if self.dataset is not None:
            return self.dataset.get_attributes()
        else:
            # TODO: Control de errores
            return 0

    def add_instance(self, instance):
        """

        Parameters
        ----------
        key : string
            Key of the bag
        values : ndarray
            Values of the instance to be inserted

        """
        # TODO: Check same length

        self.data = np.vstack((self.data, instance.data))

    def add_attribute(self, name, position, value=0):
        if position is None:
            position = len(self.data)
        pass

    # TODO: Terminar funcion

    def set_attribute(self, index, attribute, value):
        """
        Update value from attributes

        Parameters
        ----------
        key : string
            Bag key of the dataset

        attribute: string
            Attribute of the dataset

        value: float
            New value for the update
        """
        self.data[index][attribute] = value

    def set_attribute(self, name, value):
        pass

    def delete_instance(self, index):
        pass

    def delete_attribute(self, position):
        pass

    def delete_attribute(self, name):
        pass

    def set_dataset(self,dataset):
        self.dataset=dataset

    def show_bag(self):
        # TODO: Check

        table = [[self.key] + self.get_attributes()]
        count = 0
        for instance in self.data:
            table.append([count] + list(instance))
            count += 1
        # table = [['col 1', 'col 2', 'col 3', 'col 4'], [1, 2222, 30, 500], [4, 55, 6777, 1]]
        # print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        # print(tabulate([key], tablefmt="grid"))
        print(tabulate(table, headers='firstrow', tablefmt="grid", numalign="center"))
