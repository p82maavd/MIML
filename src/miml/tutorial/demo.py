from classifier.mimlTOmi.mi_classifier import MIClassifier
from classifier.mimlTOml.knn_classifier import KNNClassifier
from classifier.mimlTOml.ml_classifier import MLClassifier
from classifier.mimlTOmi.c45_classifier import C45Classifier
from classifier.mimlTOmi.decision_tree_classifier import DTClassifier

from datasets.load_dataset import load_dataset

dataset_train = load_dataset("../datasets/miml_birds_random_80train.arff", delimiter="'")
dataset_test = load_dataset("../datasets/miml_birds_random_20test.arff", delimiter="'")

classifier = MLClassifier(KNNClassifier(k=5), transformation="arithmetic")
classifier.fit(dataset_train)
classifier.evaluate(dataset_test)

#classifier = MIClassifier(DTClassifier())
#classifier.fit(dataset_train)
#classifier.evaluate(dataset_test)
