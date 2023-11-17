from datasets.load_dataset import load_dataset

dataset = load_dataset("../datasets/toy.arff")
dataset.show_dataset(head=5)
print(dataset.get_instance("bag2", 0))
dataset.describe()

#dataset.show_dataset()
#dataset.describe()
#arithmetic_transformation = ArithmeticTransformation(dataset)
#X, Y = arithmetic_transformation.transform_dataset()
