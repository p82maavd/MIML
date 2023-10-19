class MIMLDataset:
    def __init__(self) -> None:
        self.name = "undefined"
        self.attributes = []
        self.data = dict()
        self.numberlabels = 0

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_attributes(self, attributes):
        self.attributes = attributes

    def get_attributes(self):
        return self.attributes

    def set_number_labels(self, labels):
        self.numberlabels = labels

    def get_number_labels(self):
        return self.numberlabels

    def add_bag(self, key, values, labels):
        self.data[key] = (values, labels)

    def get_bag(self, key):
        #TODO
        pass

    def get_number_bags(self):
        #TODO
        pass

    def set_attribute(self,key,attribute,value):
        #TODO
        pass

    def show_dataset(self):
        # TODO: Improve this to table style
        for keys, values in self.data.items():
            print(keys)
            print(values)
