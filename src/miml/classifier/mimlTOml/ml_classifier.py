

from classifier.abstract_classifier import *
from transformation.mimlTOml.arithmetic import ArithmeticTransformation
from transformation.mimlTOml.geometric import GeometricTransformation
from transformation.mimlTOml.minmax import MinMaxTransformation


class MLClassifier(AbstractClassifier):

    def __init__(self, classifier, transformation="arithmetic"):
        super().__init__()
        transformations = ["arithmetic", "geometric", "minmax"]
        if transformation.lower() not in transformations:
            raise Exception("Specified transformation is not valid. Possible options are {arithmetic, geometric, "
                            "minmax}")
        else:
            self.transformation = transformation
        self.classifier = classifier

    def fit(self, dataset_train):
        if self.transformation == "arithmetic":
            arithmetic_transformation_train = ArithmeticTransformation(dataset_train)
            x_train, y_train = arithmetic_transformation_train.transform_dataset()
            self.classifier.fit(x_train, y_train)
        if self.transformation == "arithmetic":
            geometric_transformation_train = GeometricTransformation(dataset_train)
            x_train, y_train = geometric_transformation_train.transform_dataset()
            self.classifier.fit(x_train, y_train)
        if self.transformation == "arithmetic":
            minmax_transformation_train = MinMaxTransformation(dataset_train)
            x_train, y_train = minmax_transformation_train.transform_dataset()
            self.classifier.fit(x_train, y_train)

    def evaluate(self, dataset_test):
        arithmetic_transformation_test = ArithmeticTransformation(dataset_test)
        x_test, y_test = arithmetic_transformation_test.transform_dataset()
        self.classifier.evaluate(x_test, y_test)

