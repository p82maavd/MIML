from classifier.mimlTOmi.apr_classifier import APRClassifier
from classifier.mimlTOmi.miles_classifier import MILESClassifier
from classifier.mimlTOmi.miml_to_mi_br_classifier import MIMLtoMIBRClassifier
from classifier.mimlTOmi.miml_to_mi_classifier import MIMLtoMIClassifier
from classifier.mimlTOml.knn_classifier import KNNClassifier
from classifier.mimlTOml.miml_to_ml_classifier import MIMLtoMLClassifier

from datasets.load_dataset import load_dataset
from transformation.mimlTOml.arithmetic import ArithmeticTransformation
from transformation.mimlTOml.geometric import GeometricTransformation
from transformation.mimlTOml.minmax import MinMaxTransformation

dataset_train = load_dataset("../datasets/miml_birds_random_80train.arff", delimiter="'")
dataset_test = load_dataset("../datasets/miml_birds_random_20test.arff", delimiter="'")

#classifier = MIMLtoMLClassifier(KNNClassifier(k=5), MinMaxTransformation())
#classifier.fit(dataset_train)
#classifier.evaluate(dataset_test)

classifier = MIMLtoMIBRClassifier(APRClassifier())
classifier.fit(dataset_train)
#print(classifier.predict_bag(dataset_test.get_bag("354")))
#print(dataset_test.get_bag("354").get_labels()[0])
classifier.evaluate(dataset_test)
