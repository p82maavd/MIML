from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, \
    recall_score, accuracy_score, hamming_loss
from data.bag import Bag
from data.miml_dataset import MIMLDataset


class MIMLClassifier(ABC):
    """
    Class to represent an abstract classifier
    """

    def __init__(self):
        """
        Constructor of the class MIMLClassifier
        """

    def fit(self, dataset_train: MIMLDataset):
        """

        Parameters
        ----------
        dataset_train
        """
        if not isinstance(dataset_train, MIMLDataset):
            raise Exception("Fit function should receive a MIMLDataset as parameter")

        self.fit_internal(dataset_train)

    @abstractmethod
    def fit_internal(self, dataset_train: MIMLDataset):
        pass

    @abstractmethod
    def predict_bag(self, bag: Bag):
        """

        Parameters
        ----------
        bag
        """
        if not isinstance(bag, Bag):
            raise Exception("Predict function should receive a Numpy array as parameter")

    @abstractmethod
    def evaluate(self, dataset_test: MIMLDataset):
        """

        Parameters
        ----------
        dataset_test
        """
        if not isinstance(dataset_test, MIMLDataset):
            raise Exception("Evaluate function should receive a MIMLDataset as parameter")

