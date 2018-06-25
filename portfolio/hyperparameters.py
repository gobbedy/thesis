class Hyperparameters:

    def __init__(self, k, smoother, upper_diag, bandwidth):
        
        self.k = k
        self.smoother = smoother
        self.upper_diag = upper_diag
        self.bandwidth = bandwidth

        # see smoother.py for explanation of these lines
        self.__class__ = type(self.__class__.__name__, (self.__class__,), {})
        #self.__class__.__call__ = self.weighter_fcn

    # see smoother.py for explanation of these lines
    def __str__(self):
        return str(self.smoother) + " smoother with bandwidth " + str(self.bandwidth)

    # enable comparing two objects
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__

    #def weighter_fcn(self, x1, x2):
    #    return self.smoother(self.distance(x1, x2)/self.bandwidth)
