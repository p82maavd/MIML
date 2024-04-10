from abc import ABC, abstractmethod


class MIMLtoML(ABC):

    def __init__(self):
        self.dataset = None

    @abstractmethod
    def transform_dataset(self, dataset):
        pass

    @abstractmethod
    def transform_instance(self, key):
        pass
