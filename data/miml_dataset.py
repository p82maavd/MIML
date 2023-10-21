class MIMLDataset:
    def __init__(self) -> None:
        self.name = "undefined"
        self.attributes = []
        self.labels = []
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

    def get_number_attributes(self):
        return len(self.attributes)

    def set_labels(self, labels):
        self.labels = labels

    def get_labels(self):
        return self.labels

    def get_number_labels(self):
        return len(self.labels)

    def add_bag(self, key, values, labels):
        self.data[key] = (values, labels)

    def get_bag(self, key):
        #TODO: Formatearlo para que se vea bonito
        return self.data[key]

    def get_number_bags(self):
        return len(self.data)

    def set_attribute(self,key,attribute,value):
        #TODO
        pass

    def show_dataset(self):
        # TODO: Formatearlo para que se vea bonito
        print("Name: ", self.get_name())
        print("Attributes: ", self.get_attributes())
        print("Labels: ", self.get_labels())
        print("Bags:")
        for key in self.data:
            print("\n")
            bag = self.get_bag(key)
            print("Key: ", key)
            print("Attributes: ", bag[0])
            print("Labels: ", bag[1])
