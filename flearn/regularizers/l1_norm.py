import numpy as np

class L1_Norm(object):
    def __init__(self, coeff=1.):
        self.coeff = coeff

    def func_eval(self, x):
        return self.coeff*np.sum(np.abs(x))

    def prox_eval(self, x, prox_param):

        prox_param *= self.coeff
        return np.sign( x ) * np.maximum( np.abs( x ) - prox_param, 0 )
