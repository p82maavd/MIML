class MIMLDataset:
    """"
    Class to manage MIML data obtained from datasets
    """

    def __init__(self) -> None:

        # TODO: Si dataset leido en csv, el nombre poner el del archivo
        self.name = "undefined"
        self.attributes = []
        self.labels = []
        self.data = dict()
        self.numberlabels = 0

    def set_name(self, name):
        """
        Set function for dataset name

        Parameters
        ----------
        name : string
            Name of the dataset
        """
        self.name = name

    def get_name(self):
        """
         Get function for dataset name

        Returns
        ----------
        name : string
            Name of the dataset
        """
        return self.name

    def set_attributes(self, attributes):
        """
        Set function for dataset attributes name

        Parameters
        ----------
        attributes : List of string
            List of the attributes name of the dataset
        """
        self.attributes = attributes

    def get_attributes(self):
        """
        Get function for dataset attributes name

        Returns
        ----------
        attributes : List of strings
            Attributes name of the dataset
        """
        return self.attributes

    def get_number_attributes(self):
        """
        Get numbers of attributes of the dataset

        Returns
        ----------
         numbers of attributes: int
            Numbers of attributes of the dataset
        """
        return len(self.attributes)

    def set_labels(self, labels):
        """
        Set function for dataset labels name

        Parameters
        ----------
        labels: List of string
            List of the labels name of the dataset
        """
        self.labels = labels

    def get_labels(self):
        """
        Get function for dataset labels name

        Returns
        ----------
        labels : List of strings
            Labels name of the dataset
        """
        return self.labels

    def get_number_labels(self):
        """
        Get numbers of labels of the dataset

        Returns
        ----------
        numbers of labels: int
            Numbers of labels of the dataset
        """
        return len(self.labels)

    def get_bag(self, key):
        """
        Get data of a bag of the dataset

        Returns
        ----------
        bag: ndarray
            Attributes and labels of a bag of the dataset
        """
        # TODO: Formatearlo para que se vea bonito
        # TODO: Hacerlo quizas en funcion print_bag
        return self.data[key]

    def add_bag(self, key, values, labels):
        """
        Add a bag to the dataset

        Parameters
        ----------
        key : string
            Key of the bag

        values: ndarray
            Values of attributes of the bag

        labels: ndarray
            Labels of the bag
        """
        self.data[key] = (values, labels)

    def get_instance(self, key, index):
        """

        Parameters
        ----------
        key : string
            Key of the bag
            
        index : int
            Index of the instance in the bag

        Returns
        -------
        instance : tuple of ndarrays
            Tuple with attribute values and label of the instance

        """
        # TODO: check
        return (self.data[key][0][index], self.data[key][1])

    def add_instance(self, key, values):
        """

        Parameters
        ----------
        key : string
            Key of the bag
        values : ndarray
            Values of the instance to be inserted

        """
        # TODO: Ver si se puede hacer con las tuplas, sino ver si hay algun problema por cambiarlas a lista
        instance = (values, self.get_bag(key)[1])
        # TODO: Ver como incluirlo en la tupla actual

    def get_number_bags(self):
        """
        Get numbers of bags of the dataset

        Returns
        ----------
        numbers of bags: int
            Numbers of bags of the dataset
        """
        return len(self.data)

    def get_number_instances(self, key):
        """
        Get numbers of instances of a bag

        Parameters
        ----------
        key : string
            Key of the bag

        Returns
        ----------
        numbers of instances: int
            Numbers of instances of a bag
        """
        return len(self.get_bag(key)[0])

    def add_attribute(self, attribute, default_value=0):
        """

        Parameters
        ----------
        attribute : string
            Name of the attribute to be added
            
        default_value : int
            Default value to be inserted in new attribute column as placeholder

        """
        # TODO: Implementarlo

    def set_attribute(self, key, attribute, value):
        """
        Update value from attributes

        Parameters
        ----------
        key : string
            Bag key of the dataset

        attribute: string
            Attribute of the dataset

        value: float
            New value for the update
        """
        # TODO: Implementarlo
        pass

    def show_dataset(self):
        """"
        Function to show information about the dataset
        """
        # TODO: Formatearlo para que se vea bonito
        # TODO: Hacer algo como head y tail de pandas, ponerlo como parametro quizas
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
            # print(bag)

    def cardinality(self):
        """
        Computes the Cardinality as the average number of labels per pattern.

        Returns
        ----------
        cardinality : float
            Average number of labels per pattern
        """
        suma = 0
        for key in self.data:
            suma += sum(self.data[key][1])
        return suma / len(self.data)

    def density(self):
        """
        Computes the density as the cardinality / numLabels.

        Returns
        ----------
        density : float
            Cardinality divived by number of labels
        """
        return self.cardinality() / self.get_number_labels()

    def distinct(self):
        """
        Computes the numbers of labels combinations used in the dataset respect all the possible ones
        
        Returns
        -------
        distinct : float
            Numbers of labels combinations used in the dataset divided by all possible combinations
        """
        options = set()
        for key in self.data:
            options.add(tuple(self.data[key][1]))
        return len(options)

    def get_statistics(self):
        n_instances = 0
        max_instances = 0
        # TODO: Poner infinito
        min_instances = 100
        for key in self.data:
            instances_bag = len(self.data[key][1])
            n_instances += instances_bag
            if instances_bag < min_instances:
                min_instances = instances_bag
            elif instances_bag > max_instances:
                max_instances = instances_bag
        return n_instances, min_instances, max_instances

    def describe(self):
        """
        Print statistics about the dataset
        """

        print("-----MULTILABEL-----")
        print("Cardinalidad: ", self.cardinality())
        print("Densidad: ", self.density())
        print("Distinct: ", self.distinct())
        print("")
        # TODO: Testearlo
        n_instances, min_instances, max_instances = self.get_statistics()
        print("-----MULTIINSTANCE-----")
        print("Nº of bags; ", self.get_number_bags())
        print("Total instances: ", n_instances)
        print("Average Instances per bag: ", n_instances / self.get_number_bags())
        print("Min Instances per bag: ", min_instances)
        print("Max Instances per bag: ", max_instances)
        # TODO: Mirar bien que cuenta
        # sb.append("\nAttributesPerBag: " + attributesPerBag);
        # sb.append("\nDistribution of bags <nBags, nInstances>:");
