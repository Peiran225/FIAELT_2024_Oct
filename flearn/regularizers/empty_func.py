import numpy as np

class Empty_Function(object):
    def __init__(self, coeff=1.):
        self.coeff = coeff

    def func_eval(self, x):
        return 0

    def prox_eval(self, x, prox_param):
        return x
