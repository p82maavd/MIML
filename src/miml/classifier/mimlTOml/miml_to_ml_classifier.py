from classifier.miml_classifier import *

from transformation.mimlTOml.miml_to_ml import MIMLtoML


class MIMLtoMLClassifier(MIMLClassifier):

    def __init__(self, classifier, transformation: MIMLtoML):
        """
        Constructor of the class MIMLtoMIClassifier

        Parameters
        ----------
        classifier
            Specific classifier to be used

        transformation : MIMLtoML
            Transformation to be used
        """
        super().__init__()
        self.classifier = classifier
        self.transformation = transformation

    def fit(self, dataset_train: MIMLDataset):
        """

        Parameters
        ----------
        dataset_train
        """
        super().fit(dataset_train)
        x_train, y_train = self.transformation.transform_dataset(dataset_train)
        self.classifier.fit(x_train, y_train)

    def predict(self, data_test: np.ndarray):
        """

        Parameters
        ----------
        data_test
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
        x_test, y_test = self.transformation.transform_dataset(dataset_test)
        self.classifier.evaluate(x_test, y_test)
