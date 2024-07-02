import numpy as np

class L2_Norm(object):
    def __init__(self, coeff=1.):
        self.coeff = coeff

    def func_eval(self, x):
        return self.coeff*np.sum(np.square(x))

    def prox_eval(self, x, prox_param):
        prox_param *= self.coeff
        norm_x = np.linalg.norm(x, ord=2)
        return np.maximum(1 - prox_param/norm_x, 0) * x