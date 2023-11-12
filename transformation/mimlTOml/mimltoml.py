import numpy as np

from data.miml_dataset import MIMLDataset


class MultilabelTransformation:

    def __init__(self, dataset: MIMLDataset, mode="arithmetic"):
        self.dataset = dataset
        self.mode = ""
        self.set_mode(mode)

    def transform_dataset(self):
        if self.mode == "arithmetic":
            return self.arithmetic()
        elif self.mode == "geometric":
            return self.geometric()
        elif self.mode == "minmax":
            return self.minmax()

    def set_mode(self, mode):
        modes = ["arithmetic", "geometric", "minmax"]
        if mode not in modes:
            raise ValueError("set_mode: mode must be one of %r." % modes)
        self.mode = mode

    def arithmetic(self):
        x = np.empty(shape=(len(self.dataset.data.keys()), self.dataset.get_number_attributes()))
        y = np.empty(shape=(len(self.dataset.data.keys()), self.dataset.get_number_labels()))
        count = 0
        for keys, pattern in self.dataset.data.items():
            new_instance = np.sum(pattern[0], axis=0)
            new_instance /= pattern[0].shape[0]
            x[count] = new_instance
            y[count] = pattern[1]
            count += 1

        return x, y

    def geometric(self):
        x = np.empty(shape=(len(self.dataset.data.keys()), self.dataset.get_number_attributes()))
        y = np.empty(shape=(len(self.dataset.data.keys()), self.dataset.get_number_labels()))
        count = 0
        for keys, pattern in self.dataset.data.items():
            new_instance = np.multiply(pattern[0], axis=0)
            # TODO: No funciona con valores negativos, opcion de sumar min value
            new_instance = new_instance ** (1 / pattern[0].shape[0])
            x[count] = new_instance
            y[count] = pattern[1]
            count += 1

        return x, y

    def minmax(self):
        # TODO: Primer len es get_number_bags()
        x = np.empty(shape=(len(self.dataset.data.keys()), self.dataset.get_number_attributes()))
        y = np.empty(shape=(len(self.dataset.data.keys()), self.dataset.get_number_labels()))
        count = 0
        for keys, pattern in self.dataset.data.items():
            min_values = np.min(pattern[0], axis=0)
            max_values = np.max(pattern[0], axis=0)
            new_instance = (min_values + max_values) / 2
            x[count] = new_instance
            y[count] = pattern[1]
            count += 1

        return x, y
