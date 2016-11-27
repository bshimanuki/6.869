class Model:
    def model(self, data):
        raise NotImplementedError

    def name(self):
        return self.__class__.__name__
