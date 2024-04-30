import importlib
import numpy as np

from sklearn.metrics import classification_report, hamming_loss
from ..miml_classifier import MIMLClassifier

MIMLtoMLTransformation = importlib.import_module(".miml_to_ml_transformation", package="miml.transformation.mimlTOml")
Bag = importlib.import_module(".bag", package="miml.data").Bag
MIMLDataset = importlib.import_module(".miml_dataset", package="miml.data").MIMLDataset


class MIMLtoMLClassifier(MIMLClassifier):

    def __init__(self, ml_classifier, transformation: MIMLtoMLTransformation) -> None:
        """
        Constructor of the class MIMLtoMIClassifier

        Parameters
        ----------
        ml_classifier
            Specific classifier to be used

        transformation : MIMLtoMLTransformation
            Transformation to be used
        """
        super().__init__()
        self.classifier = ml_classifier
        self.transformation = transformation

    def fit_internal(self, dataset_train: MIMLDataset) -> None:
        """
        Training the classifier

        Parameters
        ----------
        dataset_train : MIMLDataset
            Dataset to train the classifier
        """
        x_train, y_train = self.transformation.transform_dataset(dataset_train)
        self.classifier.fit(x_train, y_train)

    def predict(self, x: np.ndarray):
        """
         Predict labels of given data

         Parameters
         ----------
         x : ndarray of shape (n, n_labels)
             Data to predict their labels
         """
        return self.classifier.predict(x)

    def predict_bag(self, bag: Bag) -> np.ndarray:
        """
        Predict labels of a given bag

        Parameters
        ----------
        bag : Bag
            Bag to predict their labels
        """
        # TODO: Check number attributes of bag with dataset
        # super().predict_bag(bag)
        x_bag, _ = self.transformation.transform_bag(bag)
        x_bag = np.array(x_bag, ndmin=2)
        return self.predict(x_bag)

    def evaluate(self, dataset_test: MIMLDataset):
        """
        Evaluate the model on a test dataset

        Parameters
        ----------
        dataset_test : MIMLDataset
            Test dataset to evaluate the model on.
        """
        # super().evaluate(dataset_test)
        x_test, y_test = self.transformation.transform_dataset(dataset_test)
        results = self.predict(x_test)

        print(classification_report(dataset_test.get_labels_by_bag(), results, zero_division=0))
        print("Hamming Loss: ", hamming_loss(dataset_test.get_labels_by_bag(), results))

        # TODO: To Csv file
        # report = classification_report(y_test, y_pred, output_dict=True)
        # df = pandas.DataFrame(report).transpose()
        # df.to_csv
