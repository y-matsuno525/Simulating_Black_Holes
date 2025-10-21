import numpy as np

from parameters import Parameter

class Operator:
    def check(self): pass
    def generate(self): pass

class BdG_Hamiltonian(Operator, Parameter):
    def __init__(self, parameters):
        self.parameters = parameters

    def check(self):
        #エルミート性
        #particle-hole対称性
        pass

    def generate(self):
        H_BdG = np.zeros((2*Parameter.L, 2*Parameter.L), dtype=complex)

        return Parameter.p(2)