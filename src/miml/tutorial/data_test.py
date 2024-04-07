from data.bag import Bag
from data.instance import Instance
from data.miml_dataset import MIMLDataset

values = [2, 7, 5.09, 1, 0]
instance1 = Instance(values)
instance2 = Instance(values)
bag = Bag("bag1")
bag.add_instance(instance1)
bag.add_instance(instance2)

dataset = MIMLDataset()
dataset.set_features_name(["hola1", "hola2", "hola3"])
dataset.set_labels_name(["label", "label2"])
instance1.show_instance()
dataset.add_bag(bag)
print(instance1.get_labels())
#dataset.show_dataset()
