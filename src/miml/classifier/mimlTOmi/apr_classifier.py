import random
from copy import deepcopy

import mil.models
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.tree import DecisionTreeClassifier
from mil import *

from data.bag import Bag
import numpy as np


class APRClassifier:

    def __init__(self):
        """

        """
        self.classifier = mil.models.APR(verbose=0)

    def fit(self, x_train, y_train):
        """

        Parameters
        ----------
        x_train
        y_train
        """
        self.classifier.fit(x_train, y_train)

    def predict_bag(self, bag):
        """

        Parameters
        ----------
        bag

        Returns
        -------

        """
        bag = bag.reshape(1, bag.shape[0], bag.shape[1])
        return self.classifier.predict(bag)

    def evaluate(self, x_test, y_test):
        """

        Parameters
        ----------
        x_test
        y_test
        """
        results = np.zeros(y_test.shape)
        for i, bag in enumerate(x_test):
            result = self.predict_bag(bag)
            results[i] = result
        return results
