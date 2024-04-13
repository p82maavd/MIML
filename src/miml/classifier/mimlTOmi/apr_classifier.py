import mil.models
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.tree import DecisionTreeClassifier
from mil import *


class APRClassifier:

    def __init__(self):
        """

        """
        self.classifier = mil.models.APR(step=10, verbose=0)

    def fit(self, x_train, y_train):
        """

        Parameters
        ----------
        x_train
        y_train
        """
        self.classifier.fit(x_train, y_train)

    def predict(self, x_test):
        """

        Parameters
        ----------
        x_test

        Returns
        -------

        """
        return self.classifier.predict(x_test)

    def evaluate(self, x_test, y_test):
        """

        Parameters
        ----------
        x_test
        y_test
        """
        results = self.predict(x_test)
        accuracy = accuracy_score(y_test, results)
        print(accuracy)
        print('Hamming Loss: ', round(hamming_loss(y_test, results), 2))
