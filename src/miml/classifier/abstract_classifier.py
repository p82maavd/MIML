from abc import ABC, abstractmethod

from sklearn.metrics import hamming_loss, accuracy_score


class AbstractClassifier(ABC):

    def __init__(self):
        self.num_labels = None
        self.classifier = None

    @abstractmethod
    def fit(self, training_dataset):
        pass

    @abstractmethod
    def predict(self, test_data):
        pass

    @abstractmethod
    def evaluate(self, dataset_test):
        pass
