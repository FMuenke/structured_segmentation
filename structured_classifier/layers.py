

class Layer:
    """
    Blueprint for other layers
    """
    def __init__(self):
        pass

    def __str__(self):
        s = ""
        s += "{} - {} - {}\n".format(self.layer_type, self.name, self.clf_row)
        s += "---------------------------\n"
        for p in self.previous:
            s += "--> {}\n".format(p)
        return s