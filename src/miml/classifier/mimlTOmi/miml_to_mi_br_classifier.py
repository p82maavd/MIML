import importlib
import numpy as np

from copy import deepcopy
from sklearn.metrics import classification_report, hamming_loss

from .miml_to_mi_classifier import MIMLtoMIClassifier

BinaryRelevanceTransformation = importlib.import_module(
    ".binary_relevance_transformation", package="miml.transformation.mimlTOmi").BinaryRelevanceTransformation
Bag = importlib.import_module(".bag", package="miml.data").Bag
MIMLDataset = importlib.import_module(".miml_dataset", package="miml.data").MIMLDataset


class MIMLtoMIBRClassifier(MIMLtoMIClassifier):
    """
    Class to represent a multiinstance classifier
    """

    def __init__(self, classifier) -> None:
        """
        Constructor of the class MIMLtoMIBRClassifier

        Parameters
        ----------
        classifier
            Specific classifier to be used
        """
        super().__init__(classifier)
        self.transformation = BinaryRelevanceTransformation()
        self.classifiers = []

    def fit_internal(self, dataset_train: MIMLDataset) -> None:
        """
        Training the classifier

        Parameters
        ----------
        dataset_train: MIMLDataset
            Dataset to train the classifier
        """
        for x in range(dataset_train.get_number_labels()):
            classifier = deepcopy(self.classifier)
            self.classifiers.append(classifier)

        datasets = self.transformation.transform_dataset(dataset_train)
        for i in range(len(datasets)):
            self.classifiers[i].fit(datasets[i][0], datasets[i][1])

    def predict(self, x: np.ndarray):
        """
        Predict labels of given data

        Parameters
        ----------
        x : ndarray of shape (n, n_labels)
            Data to predict their labels
        """
        results = np.zeros((len(self.classifiers)))
        # Prediction of each label
        for i in range(len(self.classifiers)):
            results[i] = self.classifiers[i].predict(x)
        return results

    def predict_bag(self, bag: Bag):
        """
        Predict labels of a given bag

        Parameters
        ----------
        bag : Bag
            Bag to predict their labels
        """
        super().predict_bag(bag)
        bags = self.transformation.transform_bag(bag)

        return self.predict(bags[0][0])

    def evaluate(self, dataset_test: MIMLDataset):
        """

        Parameters
        ----------
        dataset_test: MIMLDataset
            Dataset to test the classifier
        """
        super().evaluate(dataset_test)

        datasets = self.transformation.transform_dataset(dataset_test)

        results = np.zeros((dataset_test.get_number_bags(), dataset_test.get_number_labels()))
        for i, bag in enumerate(datasets[0][0]):
            results[i] = self.predict(bag)

        print(classification_report(dataset_test.get_labels_by_bag(), results, zero_division=0))
        print("Hamming Loss: ", hamming_loss(dataset_test.get_labels_by_bag(), results))

        # TODO: To Csv file
        # report = classification_report(y_test, y_pred, output_dict=True)
        # df = pandas.DataFrame(report).transpose()
        # df.to_csv
