import numpy as np
import copy

class Client_PD(object):
    
    def __init__(self, id, group=None, train_data={'x':[],'y':[]}, eval_data={'x':[],'y':[]}, train_model=None, options=None):
        self.model = train_model
        self.id = id 
        self.group = group
        self.train_data = {k: np.array(v) for k, v in train_data.items()}
        self.eval_data = {k: np.array(v) for k, v in eval_data.items()}
        self.num_samples = len(self.train_data['y'])
        self.test_samples = len(self.eval_data['y'])

        if options is not None:
            self.eta = options.get('eta', 1.0)
            self.alpha = options.get('alpha', 0.9)
        else:
            self.eta = 1.0
            self.alpha = 0.9

        self.first_update = True

        self.x_0 = self.get_params()
        self.lbd_i = [np.zeros_like(x) for x in self.get_params()]
        
    def set_params(self, model_params):
        self.model.set_params(model_params)

        for i in range(len(model_params)):
            self.x_0[i] = model_params[i]

    def get_params(self):
        return self.model.get_params()

    def get_grads(self, model_len):
        return self.model.get_gradients(self.train_data, model_len)
    

    def get_raw_grads(self):
        return self.model.get_raw_gradients(self.train_data)

    def set_vzero(self,vzero):
        self.model.set_vzero(vzero)

    def solve_grad(self):
        bytes_w = self.model.size
        grads = self.model.get_gradients(self.train_data)
        comp = self.model.flops * self.num_samples
        bytes_r = self.model.size
        return ((self.num_samples, grads), (bytes_w, comp, bytes_r))

    def solve_inner(self, num_epochs=1, batch_size=10, local_optim='pgd', term_alpha=0):

        bytes_w = self.model.size
        soln, comp, iterations = self.model.solve_inner(self.train_data, num_epochs, batch_size, local_optim, term_alpha)
        bytes_r = self.model.size
        return (self.num_samples, soln), (bytes_w, comp, bytes_r), iterations

    def solve_inner_dr(self, num_epochs=1, batch_size=10, debug=False, local_optim='pgd', term_alpha=0):

        bytes_w = self.model.size

        if local_optim == 'gd':
            pass
        else:
            self.model.optimizer.set_params(self.x_0, self.model)
            self.model.optimizer.update_lbd(self.lbd_i, self.model)

        soln, comp, iterations = self.model.solve_inner(self.train_data, num_epochs, batch_size, local_optim, term_alpha)
        bytes_r = self.model.size

        local_params = self.model.get_params()
        for i in range(len(self.lbd_i)):
            self.lbd_i[i] += (local_params[i] - self.x_0[i])/self.eta
            self.x_0[i] = local_params[i] + self.eta*self.lbd_i[i]

        comp += self.model.size*2

        if debug:
            print('aaa',self.lbd_i[0], local_params[0][0][0], self.x_0[0][0][0])
        
        return (self.num_samples, self.x_0), (bytes_w, comp, bytes_r), iterations

    def train_error_and_loss(self):
        tot_correct, loss = self.model.test(self.train_data)
        return tot_correct, loss, self.num_samples


    def test(self):

        tot_correct, loss = self.model.test(self.eval_data)
        return tot_correct, self.test_samples
