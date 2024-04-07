from abc import ABC, abstractmethod

from sklearn.metrics import hamming_loss, accuracy_score


class AbstractClassifier(ABC):
    """
    Class to represent an abstract classifier
    """

    def __init__(self):
        """
        Constructor of the class AbstractClassifier
        """

    @abstractmethod
    def fit(self, training_dataset):
        """

        Parameters
        ----------
        training_dataset
        """
        pass

    @abstractmethod
    def predict(self, test_data):
        """

        Parameters
        ----------
        test_data
        """
        pass

    @abstractmethod
    def evaluate(self, dataset_test):
        """

        Parameters
        ----------
        dataset_test
        """
        pass
