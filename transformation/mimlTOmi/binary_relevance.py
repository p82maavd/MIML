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
        datasets = []
        # TODO: Optimizar
        x = np.zeros(shape=(self.dataset.get_number_bags(), self.dataset.get_number_attributes()))

        y = np.empty(shape=(self.dataset.get_number_bags(), 1))
        ys = [y*self.dataset.get_number_labels()]
        count = 0
        for keys, pattern in self.dataset.data.items():
            #print("-------------------")
            #print(pattern[0])
            #print(x[count])
            x[count] = pattern[0]
            for i in range(self.dataset.get_number_labels()):
                ys[i][count] = pattern[1][i]
            count += 1

        for i in ys:
            datasets.append([x, i])

        return datasets

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
        return 0

    # TODO: Implementarlo
