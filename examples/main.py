from data.miml_dataset import MIMLDataset
from datasets.load_dataset import *


dataset = load_dataset("datasets/toy.arff")
dataset.showDataset()