import numpy as np
from data.miml_dataset import *


def arithmetic(dataset: MIMLDataset):
    x = np.empty(shape=(len(dataset.data.keys()), dataset.get_number_attributes()))
    y = np.empty(shape=(len(dataset.data.keys()), dataset.get_number_labels()))
    count = 0
    for keys, pattern in dataset.data.items():
        # newinstance = np.empty(dataset.get_number_attributes())
        # for instance in pattern[0]:
        #    newinstance += instance
        newinstance = np.sum(pattern[0], axis=0)
        newinstance /= pattern[0].shape[0]
        # print(newinstance)
        x[count] = newinstance
        y[count] = pattern[1]
        count += 1
    # print("X Data: ",x)
    # print("Y Data: ",y)

    return x, y
