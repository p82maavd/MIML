import numpy as np
from tabulate import tabulate


class Instance:

    def __init__(self, values, classes=[]):
        self.data = np.array(values)
        # Estructura que almacene la informacion de los atributos(nombre, si es label, etc)
        # TODO: Ver si crear clase
        self.attributes = classes
        self.bag = None

    def get_number_attributes(self):
        return len(self.attributes)

    def get_number_classes(self):
        pass

    def get_attribute(self, index: int):
        return self.data.item(index)

    def get_attribute(self, attribute: str):
        index = self.attributes.index(attribute)
        return self.data.item(index)

    def set_attribute(self, index, value):
        pass

    def set_attribute(self, name, value):
        pass

    def add_attribute(self, name, position, value=0):
        if position is None:
            position = len(self.data)
        pass

    def delete_attribute(self, position):
        pass

    def delete_attribute(self, name):
        pass

    def show_instance(self):
        # TODO: Check

        table = []

        table.append(list(self.data))

        # table = [['col 1', 'col 2', 'col 3', 'col 4'], [1, 2222, 30, 500], [4, 55, 6777, 1]]
        # print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        # print(tabulate([key], tablefmt="grid"))
        print(tabulate(table, headers='firstrow', tablefmt="grid", numalign="center"))
