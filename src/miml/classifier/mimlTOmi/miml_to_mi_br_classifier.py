from copy import deepcopy

from classifier.mimlTOmi.miml_to_mi_classifier import MIMLtoMIClassifier
from classifier.miml_classifier import *
from data.miml_dataset import MIMLDataset
from transformation.mimlTOmi.binary_relevance_transformation import BinaryRelevanceTransformation


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

    def fit_internal(self, dataset_train) -> None:
        """
        Training the classifier

        Parameters
        ----------
        dataset_train: Numpy array
            Data to train the classifier
        """
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

        results_dataset = np.zeros((dataset_test.get_number_bags(), dataset_test.get_number_labels()))
        # Prediction of each label
        for i in range(dataset_test.get_number_labels()):
            results_label = np.zeros(dataset_test.get_number_bags())
            for j, bag in enumerate(datasets[i][0]):
                results_label[j] = self.classifiers[i].predict_bag(bag)
            results_dataset[:, i] = results_label.flatten()

        accuracy = accuracy_score(dataset_test.get_labels_by_bag(), results_dataset)
        average_precision = average_precision_score(dataset_test.get_labels_by_bag(), results_dataset)
        f1_macro = f1_score(dataset_test.get_labels_by_bag(), results_dataset, average='macro')
        f1_micro = f1_score(dataset_test.get_labels_by_bag(), results_dataset, average='micro')
        hamming_loss_score = hamming_loss(dataset_test.get_labels_by_bag(), results_dataset)
        precision_macro = precision_score(dataset_test.get_labels_by_bag(), results_dataset, average='macro')
        precision_micro = precision_score(dataset_test.get_labels_by_bag(), results_dataset, average='micro')
        recall_macro = recall_score(dataset_test.get_labels_by_bag(), results_dataset, average='macro')
        recall_micro = recall_score(dataset_test.get_labels_by_bag(), results_dataset, average='micro')

        print(accuracy, average_precision, f1_macro, f1_micro, hamming_loss_score, precision_macro, precision_micro,
              recall_macro, recall_micro)

