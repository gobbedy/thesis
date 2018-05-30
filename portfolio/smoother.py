import numpy as np

class Smoother:

    def __init__(self, name="Naive"):
        
        self.name=name
        
        # associate the right smoother based on the name
        switcher = {
            "Naive": self.naive_smoother,
            "Uniform": self.uniform_smoother,
            "Triangular": self.triangular_smoother,
            "Epanechnikov": self.epanechnikov_smoother,
            "Quartic": self.quartic_smoother,
            "Triweight": self.triweight_smoother,
            "Tricubic": self.tricubic_smoother,
            "Gaussian": self.gaussian_smoother,
            "Cosine": self.cosine_smoother,
            "Logistic": self.logistic_smoother,
            "Sigmoid": self.sigmoid_smoother,
        }

        # see "Best Practices: except clause"
        #     -- https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
        try:
            # make the object callable -- see https://stackoverflow.com/questions/1705928/issue-with-making-object-callable-in-python
            # ie when the object is called as a function, it now calls its smoother function
            self.__class__ = type(self.__class__.__name__, (self.__class__,), {})
            self.__class__.__call__ = switcher[name]
            
            # also make the selected smoother callable separately in case user doesn't want synctacic sugar of calling object
            self.selected_smoother = switcher[name]
        except KeyError:
            print('There is no smoother called ', name)
            raise

    # to make object printable (ie to enable print(my_smoother)
    def __str__(self):
        return self.name

    # to make smoother object comparable by string (ie to enable my_smoother == "Naive")
    # see: https://stackoverflow.com/questions/1227121/compare-object-instances-for-equality-by-their-attributes-in-python
    def __eq__(self, other): 
        return self.name == other

    def naive_smoother(self, d):
        """
            naive_smoother(d)

        # Arguments

            1. d : Distance between two points

        # Returns

            1. Sn : Smoother coefficient
        """
        return 1.0


    def uniform_smoother(self, d):
        """
            uniform_smoother(d)

        # Arguments

            1. d : Distance between two points

        # Returns

            1. Sn : Smoother coefficient
        """

        Sn = 0.0
        if abs(d) <= 1:
            Sn = 0.5

        return Sn


    def triangular_smoother(self, d):
        """
            triangular_smoother(d)

        # Arguments

            1. d : Distance between two points

        # Returns

            1. Sn : Smoother coefficient
        """

        Sn = 0
        if abs(d) <= 1:
            Sn = 1-d

        return Sn


    def epanechnikov_smoother(self, d):
        """
            epanechnikov_smoother(d)

        # Arguments

            1. d : Distance between two points

        # Returns

            1. Sn : Smoother coefficient
        """

        Sn = 0
        if abs(d) <= 1:
            Sn = 3/4*(1-d^2)

        return Sn


    def quartic_smoother(self, d):
        """
            quartic_smoother(d)

        # Arguments

            1. d : Distance between two points

        # Returns

            1. Sn : Smoother coefficient
        """

        Sn = 0
        if abs(d) <= 1:
            Sn = 15/16*(1-d**2)**2

        return Sn


    def triweight_smoother(self, d):
        """
            triweight_smoother(d)

        # Arguments

            1. d : Distance between two points

        # Returns

            1. Sn : Smoother coefficient
        """

        Sn = 0
        if abs(d) <= 1:
            Sn = 35/32*(1-d**2)**3

        return Sn


    def tricubic_smoother(self, d):
        """
            tricubic_smoother(d)

        # Arguments

            1. d : Distance between two points

        # Returns

            1. Sn : Smoother coefficient
        """
        return max(70/81*(1-d**3), 0)


    def gaussian_smoother(self, d):
        """
            gaussian_smoother(d)

        # Arguments

            1. d : Distance between two points

        # Returns

            1. Sn : Smoother coefficient
        """
        return np.exp(-d**2)/np.sqrt(2*np.pi)



    def cosine_smoother(self, d):
        """
            cosine_smoother(d)

        # Arguments

            1. d : Distance between two points

        # Returns

            1. Sn : Smoother coefficient
        """

        Sn = 0
        if abs(d) <= 1:
            Sn = np.pi/4 * cos(np.pi/2 * d)

        return Sn


    def logistic_smoother(self, d):
        """
            logistic_smoother(d)

        # Arguments

            1. d : Distance between two points

        # Returns

            1. Sn : Smoother coefficient
        """
        return 1 / (np.exp(d) + 2 + np.exp(-d))



    def sigmoid_smoother(self, d):
        """
            sigmoid_smoother(d)

        # Arguments

            1. d : Distance between two points

        # Returns

            1. Sn : Smoother coefficient
        """
        return 2/np.pi * 1/(exp(d)+exp(-d))
