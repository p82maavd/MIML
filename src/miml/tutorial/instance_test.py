from data.bag import Bag
from data.instance import Instance
from data.miml_dataset import MIMLDataset

values = [2, 7, 5.09, 1]
instance1 = Instance(values)
instance2 = Instance(values)
instance3 = Instance(values)
instance4 = Instance(values)
instance5 = Instance(values)
# print(instance1.data)
# instance1.show_instance()
bag = Bag(instance1, "bag1")
bag.add_instance(instance2)
bag.add_instance(instance3)
bag.add_instance(instance4)
bag.add_instance(instance5)
print(bag.get_number_instances())

dataset = MIMLDataset()
dataset.set_attributes(["hola1", "hola2", "hola3"])
dataset.set_labels(["label"])
dataset.add_bag(bag)
dataset.show_dataset()
