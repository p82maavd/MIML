import heapq
from abc import ABC

import numpy as np

from classifier.abstract_classifier import *


class MIMLKNNClassifier(AbstractClassifier, ABC):

    def __init__(self, dataset, num_citers=1, num_references=1, metric=None):
        super().__init__()
        self.dataset = dataset
        self.num_citers = num_citers
        self.num_references = num_references
        self.d_size = self.dataset.get_number_of_bags()
        self.metric = metric
        self.t_matrix = None
        self.phi_matrix = None
        self.weights_matrix = None

    def build_internal(self, training_set):
        if training_set is None:
            raise ValueError("training_set cannot be None")

        self.metric.set_instances(training_set)

        self.dataset = training_set
        d_size = training_set.get_num_bags()

        # Change num_references if necessary
        if d_size <= self.num_references:
            self.num_references = d_size - 1

        # Initialize matrices
        self.t_matrix = np.zeros((d_size, self.num_labels))
        self.phi_matrix = np.zeros((d_size, self.num_labels))

        self.calculate_dataset_distances()
        self.calculate_reference_matrix()

        for i in range(d_size):
            neighbours = self.get_union_neighbours(i)
            # Update matrices
            self.phi_matrix[i] = self.calculate_record_label(neighbours, i)
            self.t_matrix[i] = self.get_bag_labels(i).copy()

        self.weights_matrix = self.get_weights_matrix()

    def make_prediction_internal(self, instance):
        self.metric.update(instance)

        # Create a new distances matrix
        distance_matrix_copy = np.copy(self.distance_matrix)
        self.distance_matrix = np.zeros((self.d_size + 1, self.d_size + 1))

        for i in range(self.d_size):
            # Fill distance matrix with previous values
            self.distance_matrix[i, :self.d_size] = distance_matrix_copy[i, :self.d_size]
            # Update distance matrix with the new bag's distances
            distance = self.metric.distance(instance, self.dataset.get_bag(i))
            self.distance_matrix[i, self.d_size] = distance
            self.distance_matrix[self.d_size, i] = distance

        # Update d_size to calculate references matrix
        self.d_size += 1
        self.calculate_reference_matrix()
        # Restore d_size value
        self.d_size -= 1

        neighbours = self.get_union_neighbours(self.d_size)
        record_label = self.calculate_record_label(neighbours)

        confidences = np.zeros(self.num_labels)
        predictions = np.zeros(self.num_labels, dtype=bool)

        # Apply linear classifier to each label
        for i in range(self.num_labels):
            column = self.weights_matrix[:, i]

            decision = self.linear_classifier(column, record_label)
            predictions[i] = decision
            confidences[i] = 1.0 if decision else 0.0

        final_decision = MultiLabelOutput(predictions, confidences)
        # Restore original distance matrix
        self.distance_matrix = distance_matrix_copy

        return final_decision

    def calculate_dataset_distances(self):
        self.distance_matrix = np.zeros((self.d_size, self.d_size))
        distance = 0.0

        for i in range(self.d_size):
            first = self.dataset.get_bag(i)
            for j in range(i, self.d_size):
                second = self.dataset.get_bag(j)
                distance = self.metric.distance(first, second)
                self.distance_matrix[i, j] = distance
                self.distance_matrix[j, i] = distance

    def calculate_reference_matrix(self):
        self.ref_matrix = np.zeros((self.d_size, self.d_size), dtype=int)

        for i in range(self.d_size):
            references = self.calculate_bag_references(i)

            for j in references:
                self.ref_matrix[i, j] = 1

    def calculate_bag_references(self, index_bag):
        # Nearest neighbours of the selected bag
        nearest_neighbours = []
        # Store indices in a list, sorted by distance to the selected bag
        pq = []

        for i in range(self.d_size):
            if i != index_bag:
                heapq.heappush(pq, (self.distance_matrix[index_bag][i], i))

        # Get the R (num_references) nearest neighbours
        for _ in range(self.num_references):
            nearest_neighbours.append(heapq.heappop(pq)[1])

        return nearest_neighbours

    def get_references(self, index_bag):
        references = []
        idx = 0

        for i in range(self.d_size):
            if self.ref_matrix[index_bag][i] == 1:
                references.append(i)
                idx += 1

        return references

    def get_citers(self, index_bag):
        # Create a priority queue sorted by distance to the selected bag
        pq = []

        for i in range(self.d_size):
            if self.ref_matrix[i][index_bag] == 1:
                heapq.heappush(pq, (self.distance_matrix[i][index_bag], i))

        citers = min(self.num_citers, len(pq))
        # Nearest citers of the selected bag
        nearest_citers = [heapq.heappop(pq)[1] for _ in range(citers)]

        return nearest_citers

    def get_union_neighbours(self, index_bag):
        references = self.get_references(index_bag)
        citers = self.get_citers(index_bag)

        # Union references and citers sets
        union_set = set(references + citers)

        return list(union_set)

    def calculate_record_label(self, indices, bagIndex):
        bag = self.dataset.get_bag(bagIndex)
        label_count = [0.0] * self.dataset.get_number_labels()
        for index in indices:
            instance = bag.get_instance(index)
            for j in range(instance.get_labels()):
                if instance.get_labels()[j] == 1:
                    label_count[j] += 1

        return label_count

    def get_bag_labels(self, bag_index):
        labels = self.dataset.get_bag(bag_index).get_labels_value()
        for i in range(len(labels)):
            if labels[i] == 0:
                labels[i] = -1

        return labels

    def get_weights_matrix(self):
        t_matrix = np.array(self.t_matrix)
        phi_matrix = np.array(self.phi_matrix)
        phi_matrix_t = phi_matrix.T

        A = phi_matrix_t.dot(phi_matrix)
        B = phi_matrix_t.dot(t_matrix)

        U, S, Vt = np.linalg.svd(A)
        s_double = np.diag(S)
        threshold = 10e-10

        for i in range(len(s_double)):
            value = s_double[i, i]
            if value < threshold:
                s_double[i, i] = 0
            else:
                s_double[i, i] = 1.0 / value

        s_inv = np.linalg.inv(s_double)
        inverse_a = Vt.T.dot(s_inv).dot(U.T)

        solution = inverse_a.dot(B)

        return solution

    def linear_classifier(self, weights, record):
        decision = 0.0
        # Multiply element by element
        for i in range(self.num_labels):
            decision += weights[i] * record[i]

        return decision > 0.3
