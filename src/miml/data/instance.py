import numpy as np
from tabulate import tabulate


class Instance:

    def __init__(self, values, attributes: dict = None):
        self.dataset = None
        self.data = np.array(values)
        # TODO: Estructura que almacene la informacion de los atributos(nombre, si es label, etc)

    def get_number_attributes(self):
        if self.dataset is not None:
            return self.dataset.get_number_attributes()
        else:
            # TODO: Control de errores
            return 0

    def get_attributes(self):
        if self.dataset is not None:
            return self.dataset.get_attributes()
        else:
            # TODO: Control de errores
            return 0

    def get_number_classes(self):
        if self.dataset is not None:
            return self.dataset.get_number_classes()
        else:
            # TODO: Control de errores
            return 0

    def get_classes(self):
        if self.dataset is not None:
            return self.dataset.get_classes()
        else:
            # TODO: Control de errores
            return 0

    def get_attribute_by_index(self, index: int):
        return self.data.item(index)

    def get_attribute_by_name(self, attribute: str):
        index = list(self.dataset.get_attributes()).index(attribute)
        return self.data.item(index)


    def set_attribute_by_index(self, index: int, value):
        self.data[index] = value

    def set_attribute_by_name(self, attribute: str, value):
        index = list(self.get_attributes()).index(attribute)
        self.data[index] = value

    def set_dataset(self, dataset):
        self.dataset = dataset

    def add_attribute(self, value=0, position=None):
        if self.dataset is None:
            if position is None:
                position = len(self.data)
            self.data = np.insert(self.data, position, value)
        else:
            # TODO: Control de errores
            pass

    def delete_attribute(self, position):
        # TODO: Check
        if self.dataset is None:
            self.data = np.delete(self.data,position)
        else:
            # TODO: Control de errores
            pass

    def show_instance(self):
        # TODO: Check

        table = [list(self.data)]

        # table = [['col 1', 'col 2', 'col 3', 'col 4'], [1, 2222, 30, 500], [4, 55, 6777, 1]]
        # print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        # print(tabulate([key], tablefmt="grid"))
        print(tabulate(table, tablefmt="grid", numalign="center"))
