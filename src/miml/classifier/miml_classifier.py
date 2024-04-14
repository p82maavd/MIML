from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import hamming_loss, accuracy_score

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
        self.model_trained = False

    @abstractmethod
    def fit(self, dataset_train: MIMLDataset):
        """

        Parameters
        ----------
        dataset_train
        """
        if not isinstance(dataset_train, MIMLDataset):
            raise Exception("Fit function should receive a MIMLDataset as parameter")
        self.model_trained = True

    @abstractmethod
    def predict_bag(self, bag: Bag):
        """

        Parameters
        ----------
        bag
        """
        if not isinstance(bag, Bag):
            raise Exception("Predict function should receive a Numpy array as parameter")
        if not self.model_trained:
            raise Exception("The model has not been trained")

    @abstractmethod
    def evaluate(self, dataset_test: MIMLDataset):
        """

        Parameters
        ----------
        dataset_test
        """
        if not isinstance(dataset_test, MIMLDataset):
            raise Exception("Evaluate function should receive a MIMLDataset as parameter")
        if not self.model_trained:
            raise Exception("The model has not been trained")
