import numpy as np
from data.miml_dataset import *


def minmax(dataset: MIMLDataset):
    x = np.empty(shape=(len(dataset.data.keys()), dataset.get_number_attributes()))
    y = np.empty(shape=(len(dataset.data.keys()), dataset.get_number_labels()))
    count = 0
    for keys, pattern in dataset.data.items():
        newinstance = np.empty(pattern[0][0].shape[0])
        # for i in range(dataset.get_number_attributes()):
        #    newinstance += instance
        min = np.min(pattern[0], axis=0)
        max = np.max(pattern[0], axis=0)
        newinstance = (min+max)/2
        # print(newinstance)
        x[count] = newinstance
        y[count] = pattern[1]
        count += 1
    # print("X Data: ",x)
    # print("Y Data: ",y)

    return x, y
