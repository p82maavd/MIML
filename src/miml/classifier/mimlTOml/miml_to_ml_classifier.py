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

    def predict_bag(self, bag: Bag):
        """

        Parameters
        ----------
        bag
        """
        # TODO: Check number attributes of bag with dataset
        super().predict_bag(bag)
        x_bag, _ = self.transformation.transform_bag(bag)
        return self.classifier.predict_bag(x_bag)

    def evaluate(self, dataset_test: MIMLDataset):
        """

        Parameters
        ----------
        dataset_test
        """
        super().evaluate(dataset_test)
        x_test, y_test = self.transformation.transform_dataset(dataset_test)
        self.classifier.evaluate(x_test, y_test)

