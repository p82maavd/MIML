import numpy as np
from collections import Counter


class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for binary splitting
        self.value = value  # Value if the node is a leaf
        self.left = left  # Left child
        self.right = right  # Right


class C45Classifier:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.min_samples_split = min_samples_split  # Minimum number of samples required to split a node
        self.max_depth = max_depth  # Maximum depth of the tree
        self.tree = None

    def fit(self, x, y):
        self.tree = self.build_tree(x, y)

    def build_tree(self, x, y, depth=0):
        num_classes = len(np.unique(y))

        # Base cases
        if depth >= self.max_depth or num_classes == 1 or x.shape[0] < self.min_samples_split:
            return DecisionNode(value=Counter(y.flatten()).most_common(1)[0][0])

        # Calculate information gain for each feature
        information_gains = []
        for feature in range(x.shape[1]):
            unique_values = np.unique(x[:, feature])
            for value in unique_values:
                left_indices = np.where(x[:, feature] <= value)[0]
                right_indices = np.where(x[:, feature] > value)[0]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue  # Skip if one of the splits is empty
                gain = self.information_gain(y, left_indices, right_indices)
                information_gains.append((feature, value, gain))

        if not information_gains:
            return DecisionNode(value=Counter(y).most_common(1)[0][0])

        # Select the best feature and value to split on
        best_feature, best_value, _ = max(information_gains, key=lambda z: z[2])

        # Split the data
        left_indices = np.where(x[:, best_feature] <= best_value)[0]
        right_indices = np.where(x[:, best_feature] > best_value)[0]

        # Recur on the sublists obtained by splitting
        left_subtree = self.build_tree(x[left_indices, :], y[left_indices], depth + 1)
        right_subtree = self.build_tree(x[right_indices, :], y[right_indices], depth + 1)

        return DecisionNode(feature_index=best_feature, threshold=best_value, left=left_subtree, right=right_subtree)

    def information_gain(self, y, left_indices, right_indices):
        left_entropy = self.calculate_entropy(y[left_indices])
        right_entropy = self.calculate_entropy(y[right_indices])
        parent_entropy = self.calculate_entropy(y)
        left_weight = len(left_indices) / len(y)
        right_weight = len(right_indices) / len(y)
        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def calculate_entropy(self, y, binary_classification=True):

        entropy = 0
        total_instances = len(y)

        if binary_classification:
            total_sum = int(np.sum(y))
            probability0 = (total_instances - total_sum) / total_instances
            probability1 = total_sum / total_instances
            # print("Prob 1: ", probability1)
            if probability0 == 0:
                probability0 = 1 / total_instances
            entropy -= probability0 * np.log2(probability0)
            if probability1 == 0:
                probability1 = 1 / total_instances
            entropy -= probability1 * np.log2(probability1)

        if not binary_classification:
            counts = Counter(y.flatten())
            for label in counts:
                probability = counts[label] / total_instances
                entropy -= probability * np.log2(probability)

        return entropy

    def predict(self, x):
        return [self.predict_instance(instance, self.tree) for instance in x]

    def predict_instance(self, instance, node):
        if node.value is not None:
            return node.value
        if instance[node.feature_index] <= node.threshold:
            return self.predict_instance(instance, node.left)
        else:
            return self.predict_instance(instance, node.right)
