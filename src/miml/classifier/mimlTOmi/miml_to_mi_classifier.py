
from classifier.miml_classifier import *
from data.miml_dataset import MIMLDataset


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

    @abstractmethod
    def fit_internal(self, dataset_train: MIMLDataset):
        pass

    def predict_bag(self, bag: Bag):
        """
        Predict labels of given data

        Parameters
        ----------
        bag : Bag
            Bag to predict their classes
        """
        super().predict_bag(bag)

    def evaluate(self, dataset_test: MIMLDataset):
        """

        Parameters
        ----------
        dataset_test
        """
        super().evaluate(dataset_test)
