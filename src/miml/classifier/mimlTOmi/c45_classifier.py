from sklearn.metrics import accuracy_score, hamming_loss
from collections import Counter
import numpy as np


class C45Classifier:

    def __init__(self):
        self.tree = None

    def fit(self, x_train, y_train):
        self.tree = self.build_tree(x_train, y_train)

    def predict(self, x_test):
        predictions = []
        for instance in x_test:
            predictions.append(self.predict_instance(instance, self.tree))
        return predictions

    def build_tree(self, x_train, y_train):
        if len(set(y_train)) == 1:  # If all labels are the same, return leaf node
            return y_train[0]
        if len(x_train) == 0:  # If there are no features left, return the majority label
            return Counter(y_train).most_common(1)[0][0]

        best_split_feature, best_split_value = self.find_best_split(x_train, y_train)
        if best_split_feature is None:  # If we can't split anymore, return the majority label
            return Counter(y_train).most_common(1)[0][0]

        left_indices = x_train[:, best_split_feature] <= best_split_value
        right_indices = ~left_indices

        left_tree = self.build_tree(x_train[left_indices], y_train[left_indices])
        right_tree = self.build_tree(x_train[right_indices], y_train[right_indices])

        return (best_split_feature, best_split_value, left_tree, right_tree)

    def find_best_split(self, x_train, y_train):
        best_split_feature = None
        best_split_value = None
        best_information_gain = -1

        for feature in range(x_train.shape[1]):
            values = np.unique(x_train[:, feature])
            for value in values:
                left_indices = x_train[:, feature] <= value
                right_indices = ~left_indices
                if len(y_train[left_indices]) == 0 or len(y_train[right_indices]) == 0:
                    continue

                information_gain = self.calculate_information_gain(y_train, y_train[left_indices], y_train[right_indices])
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_split_feature = feature
                    best_split_value = value

        return best_split_feature, best_split_value

    def calculate_entropy(self, y_train):
        counts = Counter(y_train)
        entropy = 0
        total_instances = len(y_train)
        for label in counts:
            probability = counts[label] / total_instances
            entropy -= probability * np.log2(probability)
        return entropy

    def calculate_information_gain(self, parent, left_child, right_child):
        entropy_parent = self.calculate_entropy(parent)
        total_instances = len(left_child) + len(right_child)
        weight_left = len(left_child) / total_instances
        weight_right = len(right_child) / total_instances
        entropy_children = (weight_left * self.calculate_entropy(left_child) +
                            weight_right * self.calculate_entropy(right_child))
        information_gain = entropy_parent - entropy_children
        return information_gain

    def predict_instance(self, instance, tree):
        if isinstance(tree, str):
            return tree
        feature, value, left_tree, right_tree = tree
        if instance[feature] <= value:
            return self.predict_instance(instance, left_tree)
        else:
            return self.predict_instance(instance, right_tree)
