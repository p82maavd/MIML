
import numpy as np

from classifier.abstract_classifier import *
from transformation.mimlTOmi.binary_relevance import BinaryRelevanceTransformation


class MIClassifier(AbstractClassifier):

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.classifiers = []

    def fit(self, dataset_train):
        self.classifiers = [self.classifier]*dataset_train.get_number_labels()
        binary_relevance_transformation_train = BinaryRelevanceTransformation(dataset_train)
        datasets = binary_relevance_transformation_train.transform_dataset()
        for i in range(len(datasets)):
            self.classifiers[i].fit(datasets[i][0], datasets[i][1])

    def predict(self, test_data):
        self.classifier.predict(test_data)

    def evaluate(self, dataset_test):
        binary_relevance_transformation_test = BinaryRelevanceTransformation(dataset_test)
        datasets = binary_relevance_transformation_test.transform_dataset()
        results = []
        for i in range(len(datasets)):
            results.append(self.classifiers[i].predict(datasets[i][0], datasets[i][1]))
        for j in range(1, len(results)):
            results[0] = np.hstack((results[0], results[j]))

        accuracy = accuracy_score(dataset_test.get_labels(), results[0])
        print(accuracy)
        print('Hamming Loss: ', round(hamming_loss(dataset_test.get_labels(), results), 2))


