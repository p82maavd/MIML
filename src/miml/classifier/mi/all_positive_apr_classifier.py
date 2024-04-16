from copy import deepcopy

import numpy as np

from data.bag import Bag


class AllPositiveAPRClassifier:

    def __init__(self) -> None:
        """

        """
        # self.classifier = mil.models.APR(step=10, verbose=0)
        self.apr = []

    def fit(self, x_train, y_train) -> None:
        """

        Parameters
        ----------
        x_train
        y_train
        """
        self.generate_apr(x_train, y_train)

    def predict_bag(self, bag: np.array) -> int:
        """

        Parameters
        ----------
        bag: np.array
            features values of a bag

        Returns
        -------

        """
        if np.all(bag >= self.apr[0]):
            if np.all(bag <= self.apr[1]):
                return 1
        return 0

    def generate_apr(self, x_train, y_train) -> None:

        positive_bag_indices = np.where(y_train == 1)[0]

        initial_bag_index = np.random.choice(positive_bag_indices)
        initial_instance_index = np.random.choice(x_train[initial_bag_index].shape[0])
        apr_min = apr_max = x_train[initial_bag_index][initial_instance_index]

        for bag_index in positive_bag_indices:
            for instance in x_train[bag_index]:
                apr_min = np.minimum(apr_min, instance)
                apr_max = np.maximum(apr_max, instance)

        self.apr = [apr_min, apr_max]
