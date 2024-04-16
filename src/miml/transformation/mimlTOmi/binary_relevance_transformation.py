import numpy as np

from data.bag import Bag
from data.miml_dataset import MIMLDataset


class BinaryRelevanceTransformation:
    """
    Class that performs a binary relevance transformation to convert a MIMLDataset class to numpy ndarrays.
    """

    def __init__(self):
        self.dataset = None

    def transform_dataset(self, dataset):
        """
        Transform the dataset to multilabel dataset converting each bag into a single instance being the value of each
        attribute the mean value of the instances in the bag.

        Returns
        -------

        X : {numpy ndarray} of shape (number of instances, number of attributes)
        Training vector

        Y : {numpy ndarray} of shape (number of instances, number of labels)
        Target vector relative to X.

        """
        self.dataset = dataset
        datasets = []
        x = self.dataset.get_features_by_bag()
        y = self.dataset.get_labels_by_bag()
        for i in range(self.dataset.get_number_labels()):
            datasets.append([x, y[:, i].reshape(-1, 1)])

        return datasets

    def transform_bag(self, bag: Bag):
        """
        Transform miml bag to multi instance bags

        Parameters
        ----------
        bag :
            Bag to be transformed to multiinstance bag

        Returns
        -------
        instance : tuple
        Tuple of numpy ndarray with attribute values and labels

        """
        bags = []
        for i in range(bag.get_number_labels()):
            bags.append([bag.get_features(), bag.get_labels()[0][i]])
        return bags


