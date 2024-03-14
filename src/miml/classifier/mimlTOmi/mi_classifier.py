
from classifier.abstract_classifier import *
from transformation.mimlTOmi.binary_relevance import BinaryRelevanceTransformation


class MIClassifier(AbstractClassifier):

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def fit(self, dataset_train):
        binary_relevance_transformation_train = BinaryRelevanceTransformation(dataset_train)
        x_train, y_train = binary_relevance_transformation_train.transform_dataset()
        self.model.fit(x_train, y_train)

    def evaluate(self, dataset_test):
        binary_relevance_transformation_train = BinaryRelevanceTransformation(dataset_test)
        x_test, y_test = binary_relevance_transformation_train.transform_dataset()
        self.classifier.evaluate(x_test, y_test)
