import numpy as np

from data.miml_dataset import MIMLDataset


class BinaryRelevanceTransformation:
    """
    Class that performs a binary relevance transformation to convert a MIMLDataset class to numpy ndarrays.
    """

    def __init__(self, dataset: MIMLDataset):
        self.dataset = dataset

    def transform_dataset(self):
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
        # TODO: Optimizar
        datasets = []
        x = self.dataset.get_features()
        y = self.dataset.get_labels()
        for i in range(len(self.dataset.get_number_labels())):
            datasets.append([x, y[0:, i]])

        return datasets

    def transform_bag(self, key):
        """
        Transform miml bag to multi instance bags

        Parameters
        ----------
        key : string
            Key of the bag to be transformed to multilabel instance

        Returns
        -------
        instance : tuple
        Tuple of numpy ndarray with attribute values and labels

        """
        bag = self.dataset.get_bag(key)
        bags = []
        for i in range(len(bag.get_number_labels())):
            bags.append([bag.get_attributes(), bag.get_labels()[i]])
        return bags

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
        bag = self.dataset.get_bag(key)
        return []

    # TODO: Implementarlo
