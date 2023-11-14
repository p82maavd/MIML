import numpy as np

from data.miml_dataset import MIMLDataset


class MultilabelTransformation:

    """
    Class that performs a transformation to convert a MIMLDataset class to numpy ndarrays.
    """
    def __init__(self, dataset: MIMLDataset, mode="arithmetic"):
        self.dataset = dataset
        self.mode = ""
        self.set_mode(mode)

    def transform_dataset(self):
        """
        Transform the dataset to multilabel dataset

        Returns
        -------

        X : {numpy ndarray} of shape (number of instances, number of attributes)
        Training vector

        Y : {numpy ndarray} of shape (number of instances, number of labels)
        Target vector relative to X.

        """
        if self.mode == "arithmetic":
            return self.arithmetic()
        elif self.mode == "geometric":
            return self.geometric()
        elif self.mode == "minmax":
            return self.minmax()

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
        

        

    def set_mode(self, mode):
        """
        Set the type of the multilabel transformation
        
        Parameters
        ----------
        
        mode : string
            Type of multilabel transformation

        """
        modes = ["arithmetic", "geometric", "minmax"]
        if mode not in modes:
            raise ValueError("set_mode: mode must be one of %r." % modes)
        self.mode = mode

    def arithmetic(self):
        """
        This transformation convert each bag into a single instance being the value of each attribute the mean value of
        the instances in the bag.

        Returns
        -------
        
        X : {numpy ndarray} of shape (number of instances, number of attributes)
        Training vector

        Y : {numpy ndarray} of shape (number of instances, number of labels)
        Target vector relative to X.

        """
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

        """
        This transformation convert each bag into a single instance being the value of each attribute the geometric mean
        value of the instances in the bag.

        Returns
        -------

        X : {numpy ndarray} of shape (number of instances, number of attributes)
        Training vector

        Y : {numpy ndarray} of shape (number of instances, number of labels)
        Target vector relative to X.

        """
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
        """
        This transformation convert each bag into a single instance being the value of each attribute the mean of the
        min and max value of the instances in the bag.

        Returns
        -------

        X : {numpy ndarray} of shape (number of instances, number of attributes)
        Training vector

        Y : {numpy ndarray} of shape (number of instances, number of labels)
        Target vector relative to X.

        """
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
