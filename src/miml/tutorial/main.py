from classifier.mimlTOml.ml_classifier import  MLClassifier
from datasets.load_dataset import load_dataset

dataset_train = load_dataset("../datasets/miml_birds_random_80train.arff", delimiter="'")
dataset_test = load_dataset("../datasets/miml_birds_random_20test.arff", delimiter="'")

classifier = MLClassifier()

classifier.fit(dataset_train)

classifier.evaluate(dataset_test)

