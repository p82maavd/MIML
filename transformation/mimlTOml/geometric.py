import numpy as np
from data.miml_dataset import *


def geometric(dataset: MIMLDataset):
    x = np.empty(shape=(len(dataset.data.keys()), dataset.get_number_attributes()))
    y = np.empty(shape=(len(dataset.data.keys()), dataset.get_number_labels()))
    count = 0
    for keys, pattern in dataset.data.items():
        # newinstance = np.empty(pattern[0][0].shape[0])
        # for i in range(len(newinstance)):
        #    newinstance[i] = 1
        # for instance in pattern[0]:
        #    newinstance *= instance
        # TODO: Testear si funciona bien
        newinstance = np.multiply(pattern[0], axis=0)
        # TODO: No funciona con valores negativos, opcion de sumar min value
        newinstance = newinstance ** (1 / pattern[0].shape[0])
        # print(newinstance)
        x[count] = newinstance
        y[count] = pattern[1]
        count += 1
    # print("X Data: ",x)
    # print("Y Data: ",y)

    return x, y
