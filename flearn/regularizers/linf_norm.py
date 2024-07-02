import numpy as np

class LInf_Norm(object):
    def __init__(self, coeff=1.):
        self.coeff = coeff

    def func_eval(self, x):
        return self.coeff*np.max(np.abs(x))

    def prox_eval(self, x, prox_param):
        prox_param *= self.coeff
        return x - prox_param * self.proj_l1_ball(x / prox_param)

    def proj_l1_ball(self, x, prox_param=1 ):
        norm_x = np.linalg.norm(x, ord=1)

        if norm_x <= 1:
            return x
        else:
            sort_x = np.sort(x)[::-1]
            tmp_sum = 0
            index = -1
            for i in range(len(sort_x)):
                tmp_sum += sort_x[i]
                if sort_x[i] <= (1.0/(i+1)) * (tmp_sum - 1):
                    break
                else:
                    index += 1

                index = np.max(index,0)

            prox_param = np.max((1.0/(index+1))*(tmp_sum - 1),0)

            return prox_l1_norm(x, prox_param)

    def prox_l1_norm(self, x, prox_param=1 ):
        return np.sign( x ) * np.maximum( np.abs( x ) - prox_param, 0 )


