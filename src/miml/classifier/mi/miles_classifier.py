import mil.models
from mil.bag_representation import MILESMapping
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from mil.models import SVC


class MILESClassifier:

    def __init__(self, sigma2=4.5 ** 2, c=0.5):
        """

        """
        self.classifier = mil.models.MILES()
        self.model = None
        self.mapping = None
        self.sigma2 = sigma2
        self.c = c
        self.trainer = None

    def fit(self, x_train, y_train):
        """

        Parameters
        ----------
        x_train
        y_train
        """

        self.classifier.check_exceptions(x_train)
        self.mapping = MILESMapping(self.sigma2)
        mapped_bags = self.mapping.fit_transform(x_train)

        # train the SVM
        # self.model = LinearSVC(penalty="l1", C=self.c, dual=False, class_weight='balanced',max_iter=100000)

        self.model = DecisionTreeClassifier()
        self.model.fit(mapped_bags, y_train.flatten())

    def predict_bag(self, bag):
        """

        Parameters
        ----------
        bag

        Returns
        -------

        """

        bag = bag.reshape(1, bag.shape[0], bag.shape[1])
        # return self.classifier.predict(bag)

        # testeo

        mapped_bags = self.mapping.transform(bag)
        return self.model.predict(mapped_bags)

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
