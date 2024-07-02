import numpy as np

class Client(object):
    
    def __init__(self, id, group=None, train_data={'x':[],'y':[]}, eval_data={'x':[],'y':[]}, model=None):
        self.model = model
        self.id = id 
        self.group = group
        self.train_data = {k: np.array(v) for k, v in train_data.items()}
        self.eval_data = {k: np.array(v) for k, v in eval_data.items()}
        self.num_samples = len(self.train_data['y'])
        self.test_samples = len(self.eval_data['y'])

    def set_params(self, model_params):
        self.model.set_params(model_params)

    def get_params(self):
        return self.model.get_params()

    def get_grads(self, model_len):
        return self.model.get_gradients(self.train_data, model_len)

    def solve_grad(self):
        bytes_w = self.model.size
        grads = self.model.get_gradients(self.train_data)
        comp = self.model.flops * self.num_samples
        bytes_r = self.model.size
        return ((self.num_samples, grads), (bytes_w, comp, bytes_r))
    
    def get_raw_grads(self):
        return self.model.get_raw_gradients(self.train_data)

    def set_vzero(self,vzero):
        self.model.set_vzero(vzero)

    def solve_inner(self, num_epochs=1, batch_size=10, local_optim='pgd', term_alpha=0):

        bytes_w = self.model.size
        soln, comp, iterations = self.model.solve_inner(self.train_data, num_epochs, batch_size, local_optim, term_alpha)
        bytes_r = self.model.size
        return (self.num_samples, soln), (bytes_w, comp, bytes_r), iterations

    def solve_iters(self, num_iters=1, batch_size=10):

        bytes_w = self.model.size
        soln, comp = self.model.solve_iters(self.train_data, num_iters, batch_size)
        bytes_r = self.model.size
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)

    def train_error_and_loss(self):
        tot_correct, loss = self.model.test(self.train_data)
        return tot_correct, loss, self.num_samples


    def test(self):
        tot_correct, loss = self.model.test(self.eval_data)
        return tot_correct, self.test_samples
