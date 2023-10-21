from datasets.load_dataset import *

dataset = load_dataset("../datasets/toy.arff")
#print(dataset.get_bag("bag1"))
dataset.show_dataset()
