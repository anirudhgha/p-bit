import random
import numpy as np


class P_circuit:
    def __init__(self, Nm=0, J=[[]], h=[]):
        self.Nm = Nm
        self.J = J
        self.h = h

    def __call__(self):
        return self.J, self.h

    def setWeights(self, J, h):
        self.J = J
        self.h = h

    def setSize(self, num_pbits):
        self.Nm = num_pbits

    def buildRandomNetwork(self, num_pbits, J_max_weight=5, h_max_weight=5, random_H=False):
        """
        build a random p-circuit 
        """
        self.Nm = num_pbits
        for i in range(self.Nm):


J = [[1, 2], [3, 4]]
Nm = 2
h = [1, 1]

pcircuit = P_circuit(Nm, J, h)
pcircuit = P_circuit()

pcircuit.setWeights(J, h)
pcircuit.setSize(4)
print(pcircuit())
