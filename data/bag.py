

class Bag:


    def __init__(self, instance):
        #Crear numpy ndarray 3d con una instancia
        self.data = []
        self.key = ""

    def __init__(self, instances):
        #Crear numpy ndarray 3d con varias instancias
        pass

    def get_number_instances(self):
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
        return len(self.data)

    def get_instance(self, index):
        """

        Parameters
        ----------

        index : int
            Index of the instance in the bag

        Returns
        -------
        instance : tuple of ndarrays
            Tuple with attribute values and label of the instance

        """
        pass

    def add_instance(self, instance):
        """

        Parameters
        ----------
        key : string
            Key of the bag
        values : ndarray
            Values of the instance to be inserted

        """
        pass

    def add_attribute(self, name, position, value=0):
        if position is None:
            position = len(self.data)
        pass

    def set_attribute(self, index, value):
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
        pass

    def set_attribute(self, name, value):
        pass

    def delete_instance(self, index):
        pass

    def delete_attribute(self, position):
        pass

    def delete_attribute(self, name):
        pass

