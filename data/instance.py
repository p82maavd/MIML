class Instance:

    def __init__(self, values, classes):
        self.data = values
        #Estructura que almacene la informacion de los atributos(nombre, si es label, etc)
        #TODO: Ver si crear clase
        self.attributes = []

    def get_number_attributes(self):
        return len(self.attributes)

    def get_number_classes(self):

    def get_attribute(self, index):
        pass

    def get_attribute(self, attribute):
        pass

    def set_attribute(self, index, value):
        pass

    def set_attribute(self, name, value):
        pass

    def add_attribute(self, name, position, value=0):
        if position is None:
            position = len(self.data)
        pass

    def delete_attribute(self, position):
        pass

    def delete_attribute(self, name):
        pass

