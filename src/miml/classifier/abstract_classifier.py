from abc import ABC, abstractmethod

from sklearn.metrics import hamming_loss, accuracy_score


class AbstractClassifier(ABC):

    def __init__(self):
        self.num_labels = None
        self.model = None

    @abstractmethod
    def fit(self, training_dataset):
        pass

    def predict(self, test_data):
        return self.model.predict(test_data)

    def evaluate(self, dataset_test):

        results = self.predict(dataset_test.get_features())
        accuracy = accuracy_score(dataset_test.get_labels(), results)
        print(accuracy)
        print('Hamming Loss: ', round(hamming_loss(dataset_test.get_labels(), results), 2))
