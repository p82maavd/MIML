from datasets.load_dataset import *
from transformation.mimlTOml.arithmetic import *
from transformation.mimlTOml.geometric import *
from transformation.mimlTOml.minmax import *

dataset = load_dataset("../datasets/toy.arff")
#print(dataset.get_bag("bag1"))
dataset.show_dataset()
dataset.describe()
print(arithmetic(dataset))
print(minmax(dataset))