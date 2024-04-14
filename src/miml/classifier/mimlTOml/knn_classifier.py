import numpy as np
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:

    def __init__(self, k=3):
        """

        Parameters
        ----------
        k
        """
        self.k = k
        self.classifier = KNeighborsClassifier(n_neighbors=self.k)

    def fit(self, x_train, y_train):
        """

        Parameters
        ----------
        x_train
        y_train
        """
        self.classifier.fit(x_train, y_train)

    def predict_bag(self, x_test):
        """

        Parameters
        ----------
        x_test

        Returns
        -------

        """
        x_test = np.array(x_test, ndmin=2)
        return self.classifier.predict(x_test)

    def evaluate(self, x_test, y_test):
        """

        Parameters
        ----------
        x_test: Numpy Array [Nº of bags, nº labels]
        y_test
        """
        results = np.zeros(y_test.shape)
        for i, bag in enumerate(x_test):
            result = self.predict_bag(bag)
            results[i] = result
        accuracy = accuracy_score(y_test, results)
        print(accuracy)
        print('Hamming Loss: ', round(hamming_loss(y_test, results), 2))
