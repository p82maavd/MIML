from datasets.load_dataset import load_dataset
from transformation.mimlTOmi.binary_relevance import BinaryRelevanceTransformation

dataset = load_dataset("../datasets/toy.csv")

binary_relevance = BinaryRelevanceTransformation(dataset)
datasets = binary_relevance.transform_dataset()

print(datasets)
for x in datasets:
    print("----------------------")
    print(x)
    print("----------------------")