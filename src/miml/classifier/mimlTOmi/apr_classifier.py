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
        # self.classifier = mil.models.APR(step=10, verbose=0)
        self.rectangles = []
        self.labels = None

    def fit(self, x_train, y_train):
        """

        Parameters
        ----------
        x_train
        y_train
        """
        self.labels = y_train
        num_classes = len(set(list(y_train.flatten())))

        # For each class, find the axis-parallel rectangle that encompasses all bags of that class
        for class_label in range(num_classes):
            class_bags = [x_train[i] for i in range(len(x_train)) if self.labels[i] == class_label]
            if len(class_bags) > 0:
                min_vals = np.min(np.vstack(class_bags), axis=0)
                max_vals = np.max(np.vstack(class_bags), axis=0)
                rectangle = (min_vals, max_vals)
                self.rectangles.append(rectangle)

    def predict_bag(self, bag):
        """

        Parameters
        ----------
        bag

        Returns
        -------

        """
        options = []
        for i, rectangle in enumerate(self.rectangles):
            min_vals, max_vals = rectangle
            if np.all(bag >= min_vals):
                if np.all(bag <= max_vals):
                    options.append(i)
        # TODO: Hacer que coga el mas cercano y que funcione no solo para binary classification
        if len(options) == 0:
            return 0
        if len(options) == 1:
            return options[0]
        if len(options) == 2:
            return options[round(random.random())]

        # return None  # Return None if no rectangle matches

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
