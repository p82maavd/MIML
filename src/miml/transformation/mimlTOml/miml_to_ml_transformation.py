import importlib

from abc import ABC, abstractmethod

Bag = importlib.import_module(".bag", package="miml.data").Bag
MIMLDataset = importlib.import_module(".miml_dataset", package="miml.data").MIMLDataset


class MIMLtoMLTransformation(ABC):

    def __init__(self) -> None:
        self.dataset = None

    @abstractmethod
    def transform_dataset(self, dataset: MIMLDataset):
        pass

    @abstractmethod
    def transform_bag(self, bag: Bag):
        pass
