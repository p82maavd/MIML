import numpy as np

from classifier.miml_classifier import *
from data.miml_dataset import MIMLDataset
from transformation.mimlTOmi.binary_relevance import BinaryRelevanceTransformation


class MIClassifier(MIMLClassifier):
    """
    Class to represent a multiinstance classifier
    """

    def __init__(self, classifier):
        """
        Constructor of the class MIClassifier

        Parameters
        ----------
        classifier
            Specific classifier to be used
        """
        super().__init__()
        self.classifier = classifier
        self.classifiers = []

    def fit(self, dataset_train):
        """
        Training the classifier

        Parameters
        ----------
        dataset_train: Numpy array
            Data to train the classifier
        """
        self.classifiers = [self.classifier] * dataset_train.get_number_labels()
        binary_relevance_transformation_train = BinaryRelevanceTransformation(dataset_train)
        datasets = binary_relevance_transformation_train.transform_dataset()
        for i in range(len(datasets)):
            self.classifiers[i].fit(datasets[i][0], datasets[i][1])

    def predict(self, test_data):
        """
        Predict labels of given data

        Parameters
        ----------
        test_data : Numpy Array
            Data to predict their classes
        """
        self.classifier.predict(test_data)

    def evaluate(self, dataset_test: MIMLDataset):
        """

        Parameters
        ----------
        dataset_test
        """
        binary_relevance_transformation_test = BinaryRelevanceTransformation(dataset_test)
        datasets = binary_relevance_transformation_test.transform_dataset()
        results = np.zeros((dataset_test.get_number_instances(), dataset_test.get_number_labels()))
        # Prediction of each label
        for i in range(len(datasets)):
            results[:, i] = self.classifiers[i].predict(datasets[i][0])

        # accuracy = accuracy_score(dataset_test.get_labels(), results[0])
        # print(accuracy)
        print('Hamming Loss: ', round(hamming_loss(dataset_test.get_labels(), results), 2))
