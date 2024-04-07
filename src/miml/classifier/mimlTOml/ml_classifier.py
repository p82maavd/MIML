
from classifier.abstract_classifier import *
from transformation.mimlTOml.arithmetic import ArithmeticTransformation
from transformation.mimlTOml.geometric import GeometricTransformation
from transformation.mimlTOml.minmax import MinMaxTransformation


class MLClassifier(AbstractClassifier):

    def __init__(self, classifier, transformation="arithmetic"):
        """
        Constructor of the class MIClassifier

        Parameters
        ----------
        classifier
            Specific classifier to be used

        transformation : String
            Type of transformation to be used
        """
        super().__init__()
        transformations = ["arithmetic", "geometric", "minmax"]
        if transformation.lower() not in transformations:
            raise Exception("Specified transformation is not valid. Possible options are {arithmetic, geometric, "
                            "minmax}")
        else:
            self.transformation = transformation
        self.classifier = classifier

    def fit(self, dataset_train):
        """

        Parameters
        ----------
        dataset_train
        """
        if self.transformation == "arithmetic":
            arithmetic_transformation_train = ArithmeticTransformation(dataset_train)
            x_train, y_train = arithmetic_transformation_train.transform_dataset()
            self.classifier.fit(x_train, y_train)
        elif self.transformation == "geometric":
            geometric_transformation_train = GeometricTransformation(dataset_train)
            x_train, y_train = geometric_transformation_train.transform_dataset()
            self.classifier.fit(x_train, y_train)
        elif self.transformation == "minmax":
            minmax_transformation_train = MinMaxTransformation(dataset_train)
            x_train, y_train = minmax_transformation_train.transform_dataset()
            self.classifier.fit(x_train, y_train)

    def predict(self, dataset_test):
        """

        Parameters
        ----------
        dataset_test
        """
        if self.transformation == "arithmetic":
            arithmetic_transformation_test = ArithmeticTransformation(dataset_test)
            x_test, y_test = arithmetic_transformation_test.transform_dataset()
            self.classifier.predict(x_test)
        elif self.transformation == "geometric":
            geometric_transformation_test = GeometricTransformation(dataset_test)
            x_test, y_test = geometric_transformation_test.transform_dataset()
            self.classifier.predict(x_test)
        elif self.transformation == "minmax":
            minmax_transformation_test = MinMaxTransformation(dataset_test)
            x_test, y_test = minmax_transformation_test.transform_dataset()
            self.classifier.predict(x_test)

    def evaluate(self, dataset_test):
        """

        Parameters
        ----------
        dataset_test
        """
        if self.transformation == "arithmetic":
            arithmetic_transformation_test = ArithmeticTransformation(dataset_test)
            x_test, y_test = arithmetic_transformation_test.transform_dataset()
            self.classifier.evaluate(x_test, y_test)
        elif self.transformation == "geometric":
            geometric_transformation_test = GeometricTransformation(dataset_test)
            x_test, y_test = geometric_transformation_test.transform_dataset()
            self.classifier.evaluate(x_test, y_test)
        elif self.transformation == "minmax":
            minmax_transformation_test = MinMaxTransformation(dataset_test)
            x_test, y_test = minmax_transformation_test.transform_dataset()
            self.classifier.evaluate(x_test, y_test)
