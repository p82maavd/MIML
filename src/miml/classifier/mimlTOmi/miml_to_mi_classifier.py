import numpy as np

from classifier.miml_classifier import *
from data.miml_dataset import MIMLDataset
from transformation.mimlTOmi.binary_relevance import BinaryRelevanceTransformation


class MIMLtoMIClassifier(MIMLClassifier):
    """
    Class to represent a multiinstance classifier
    """

    def __init__(self, classifier):
        """
        Constructor of the class MIMLtoMIClassifier

        Parameters
        ----------
        classifier
            Specific classifier to be used
        """
        super().__init__()
        self.classifier = classifier

    def fit(self, dataset_train):
        """
        Training the classifier

        Parameters
        ----------
        dataset_train: Numpy array
            Data to train the classifier
        """
        super().fit(dataset_train)

    def predict(self, data_test):
        """
        Predict labels of given data

        Parameters
        ----------
        data_test : Numpy Array
            Data to predict their classes
        """
        super().predict(data_test)
        self.classifier.predict(data_test)

    def evaluate(self, dataset_test: MIMLDataset):
        """

        Parameters
        ----------
        dataset_test
        """
        super().evaluate(dataset_test)

