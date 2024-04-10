import numpy as np

from data.miml_dataset import MIMLDataset
from transformation.mimlTOml.miml_to_ml import MIMLtoML


class ArithmeticTransformation(MIMLtoML):
    """
    Class that performs an arithmetic transformation to convert a MIMLDataset class to numpy ndarrays.
    """

    def __init__(self):
        super().__init__()

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
        x = np.empty(shape=(self.dataset.get_number_bags(), self.dataset.get_number_features()))
        y = np.empty(shape=(self.dataset.get_number_bags(), self.dataset.get_number_labels()))
        count = 0
        for key in self.dataset.data.keys():
            features, labels = self.transform_instance(key)
            x[count] = features
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
        features : numpy array
            Numpy array with feature values

        labels : numpy array
            Numpy array with label values
        """
        # TODO: Test
        features = self.dataset.get_bag(key).get_features()
        labels = self.dataset.get_bag(key).get_labels()[0]
        features = np.mean(features, axis=0)
        return features, labels
