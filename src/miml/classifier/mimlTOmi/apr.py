import numpy as np


class APRModel:
    def __init__(self):


    def fit(self, bags, labels):
        """
        Train the classifier by finding the axis-parallel rectangles
        that encompass all instances in each bag.

        Parameters:
        - bags: List of numpy arrays representing bags of instances.
        - labels: List of labels corresponding to each bag.

        """

    def predict_bag(self, test_bag):
        """
        Predict label for a test bag based on the axis-parallel rectangles.

        Parameters:
        - test_bag: Numpy array representing a bag of instances.

        Returns:
        - Predicted label.

        """
        # For each rectangle, check if the test bag falls within it


