from datasets.load_dataset import *

dataset = load_dataset("../datasets/miml_birds.arff")
#print(dataset.get_bag("bag1"))
dataset.show_dataset()
dataset.describe()