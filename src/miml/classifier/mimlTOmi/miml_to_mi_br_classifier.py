from copy import deepcopy


from classifier.mimlTOmi.miml_to_mi_classifier import MIMLtoMIClassifier
from classifier.miml_classifier import *
from data.miml_dataset import MIMLDataset
from transformation.mimlTOmi.binary_relevance_transformation import BinaryRelevanceTransformation


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
        self.transformation = BinaryRelevanceTransformation()
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
        for x in range(dataset_train.get_number_labels()):
            classifier = deepcopy(self.classifier)
            self.classifiers.append(classifier)

        datasets = self.transformation.transform_dataset(dataset_train)
        for i in range(len(datasets)):
            self.classifiers[i].fit(datasets[i][0], datasets[i][1])

    def predict_bag(self, bag: Bag):
        """
        Predict labels of given data

        Parameters
        ----------
        bag : Bag
            Bag to predict their classes
        """
        super().predict_bag(bag)
        bags = self.transformation.transform_bag(bag)
        results = np.zeros((bag.get_number_labels()))
        # Prediction of each label
        for i in range(len(bags)):
            results[i] = self.classifiers[i].predict_bag(bags[i][0])
        return results

    def evaluate(self, dataset_test: MIMLDataset):
        """

        Parameters
        ----------
        dataset_test
        """
        super().evaluate(dataset_test)

        datasets = self.transformation.transform_dataset(dataset_test)

        results = np.zeros((dataset_test.get_number_bags(), dataset_test.get_number_labels()))
        # Prediction of each label
        for i in range(dataset_test.get_number_labels()):
            results[:, i] = self.classifiers[i].evaluate(datasets[i][0], datasets[i][1]).flatten()

        accuracy = accuracy_score(dataset_test.get_labels_by_bag(), results)
        print(accuracy)
        print('Hamming Loss: ', round(hamming_loss(dataset_test.get_labels_by_bag(), results), 2))
