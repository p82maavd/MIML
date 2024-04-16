from abc import ABC, abstractmethod

from data.bag import Bag
from data.miml_dataset import MIMLDataset


class MIMLtoML(ABC):

    def __init__(self):
        self.dataset = None

    @abstractmethod
    def transform_dataset(self, dataset: MIMLDataset):
        pass

    @abstractmethod
    def transform_bag(self, bag: Bag):
        pass
