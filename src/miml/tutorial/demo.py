from classifier.mi.all_positive_apr_classifier import AllPositiveAPRClassifier
from classifier.mi.apr_classifier import APRClassifier
from classifier.mimlTOmi.miml_to_mi_br_classifier import MIMLtoMIBRClassifier
from classifier.mimlTOml.knn_classifier import KNNClassifier
from classifier.mimlTOml.miml_to_ml_classifier import MIMLtoMLClassifier
from sklearn.neighbors import KNeighborsClassifier


from datasets.load_dataset import load_dataset
from transformation.mimlTOml.minmax import MinMaxTransformation

dataset_train = load_dataset("../datasets/miml_birds_random_80train.arff", delimiter="'")
dataset_test = load_dataset("../datasets/miml_birds_random_20test.arff", delimiter="'")

#classifier = MIMLtoMLClassifier(KNeighborsClassifier(), MinMaxTransformation())
#classifier.fit(dataset_train)
#classifier.evaluate(dataset_test)

classifier = MIMLtoMIBRClassifier(APRClassifier())
classifier.fit(dataset_train)
print(classifier.predict_bag(dataset_test.get_bag("366")))
print(dataset_test.get_bag("366").get_labels()[0])
classifier.evaluate(dataset_test)
