import numpy as np

from classifier.mimlTOmi.miml_to_mi_classifier import MIMLtoMIClassifier
from classifier.miml_classifier import *
from data.miml_dataset import MIMLDataset
from transformation.mimlTOmi.binary_relevance import BinaryRelevanceTransformation


class MIMLtoMIBRClassifier(MIMLtoMIClassifier):
    """
    Class to represent a multiinstance classifier
    """

    def __init__(self, classifier):
        """
        Constructor of the class MIMLtoMIBRClassifier

        Parameters
        ----------
        classifier
            Specific classifier to be used
        """
        super().__init__(classifier)
        self.classifiers = []

    def fit(self, dataset_train):
        """
        Training the classifier

        Parameters
        ----------
        dataset_train: Numpy array
            Data to train the classifier
        """
        super().fit(dataset_train)
        self.classifiers = [self.classifier] * dataset_train.get_number_labels()
        binary_relevance_transformation_train = BinaryRelevanceTransformation(dataset_train)
        datasets = binary_relevance_transformation_train.transform_dataset()
        for i in range(len(datasets)):
            self.classifiers[i].fit(datasets[i][0], datasets[i][1])

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
        binary_relevance_transformation_test = BinaryRelevanceTransformation(dataset_test)
        datasets = binary_relevance_transformation_test.transform_dataset()
        results = np.zeros((dataset_test.get_number_bags(), dataset_test.get_number_labels()))
        # Prediction of each label
        for i in range(len(datasets)):
            results[:, i] = self.classifiers[i].predict(datasets[i][0])
        # TODO: Here should be a call to classifier evaluate. Lo que pasa es que cada apr_classifier solo se encarga de clasificar una label por lo que no puede hacer el evaluate y se va a quedar en esta clase
        accuracy = accuracy_score(dataset_test.get_labels_by_bag(), results)
        print(accuracy)
        print('Hamming Loss: ', round(hamming_loss(dataset_test.get_labels_by_bag(), results), 2))
