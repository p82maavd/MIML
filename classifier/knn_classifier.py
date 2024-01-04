from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

import classifier.abstract_classifier


class KNNClassifier(classifier.abstract_classifier.AbstractClassifier):

    def __init__(self, k=3):
        self.k = k
        self.model = None

    def train(self, training_data, training_labels):
        # Training logic using KNN
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model.fit(training_data, training_labels)

    def predict(self, test_data):
        # Prediction logic using KNN
        return self.model.predict(test_data)

    def evaluate(self, test_data, test_labels):
        # Evaluation logic using KNN
        predictions = self.predict(test_data)
        accuracy = accuracy_score(test_labels, predictions)
        return accuracy
