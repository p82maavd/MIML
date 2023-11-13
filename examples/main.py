from datasets.load_dataset import load_dataset
from transformation.mimlTOml.mimltoml import MultilabelTransformation

dataset = load_dataset("../datasets/toy.arff")
# print(dataset.get_bag("bag1"))
dataset.show_dataset()
dataset.describe()
arithmetic_transformation = MultilabelTransformation(dataset, mode="arithmetic")
X, Y = arithmetic_transformation.transform_dataset()
