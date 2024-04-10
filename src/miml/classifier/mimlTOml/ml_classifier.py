from classifier.miml_classifier import *

from transformation.mimlTOml.miml_to_ml import MIMLtoML


class MLClassifier(MIMLClassifier):

    def __init__(self, classifier, transformation: MIMLtoML):
        """
        Constructor of the class MIClassifier

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

    def fit(self, dataset_train):
        """

        Parameters
        ----------
        dataset_train
        """
        x_train, y_train = self.transformation.transform_dataset(dataset_train)
        self.classifier.fit(x_train, y_train)

    def predict(self, dataset_test):
        """

        Parameters
        ----------
        dataset_test
        """
        x_test, _ = self.transformation.transform_dataset(dataset_test)
        self.classifier.predict(x_test)

    def evaluate(self, dataset_test):
        """

        Parameters
        ----------
        dataset_test
        """
        x_test, y_test = self.transformation.transform_dataset(dataset_test)
        self.classifier.evaluate(x_test, y_test)
