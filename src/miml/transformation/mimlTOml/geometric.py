import numpy as np

from data.miml_dataset import MIMLDataset


class GeometricTransformation:
    """
    Class that performs a geometric transformation to convert a MIMLDataset class to numpy ndarrays.
    """

    def __init__(self, dataset: MIMLDataset):
        self.dataset = dataset

    def transform_dataset(self):
        """
        Transform the dataset to multilabel dataset converting each bag into a single instance being the value of each
        attribute the geometric center of the instances in the bag.

        Returns
        -------

        X : {numpy ndarray} of shape (number of instances, number of attributes)
        Training vector

        Y : {numpy ndarray} of shape (number of instances, number of labels)
        Target vector relative to X.

        """

        x = np.empty(shape=(self.dataset.get_number_bags(), self.dataset.get_number_attributes()))
        y = np.empty(shape=(self.dataset.get_number_bags(), self.dataset.get_number_labels()))
        count = 0
        for keys, bag in self.dataset.data.items():
            values = bag.data[0:, :self.dataset.get_number_attributes()]
            labels = bag.data[0, self.dataset.get_number_attributes():]
            min_values = np.min(values, axis=0)
            max_values = np.max(values, axis=0)
            new_instance = (min_values + max_values) / 2
            x[count] = new_instance
            y[count] = labels
            count += 1

        return x, y

    def transform_instance(self, key):
        """
        Transform the instances of a bag to a multilabel instance

        Parameters
        ----------
        key : string
            Key of the bag to be transformed to multilabel instance

        Returns
        -------
        instance : tuple
        Tuple of numpy ndarray with attribute values and labels

        """

    # TODO: Implementarlo
