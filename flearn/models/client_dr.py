import numpy as np
import copy

class Client_DR(object):
    
    def __init__(self, in_id, group=None, train_data={'x':[],'y':[]}, eval_data={'x':[],'y':[]}, train_model=None, options=None):

        self.model = train_model
        self.id = in_id 
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

        self.y_i = [np.zeros_like(x) for x in self.get_params()]
        self.x_hat = [np.zeros_like(x) for x in self.get_params()]
        self.zeros_model = [np.zeros_like(x) for x in self.get_params()]
        self.x_hat_diff = [np.zeros_like(x) for x in self.get_params()]
        self.latest_model = [np.zeros_like(x) for x in self.get_params()]
        
    def set_params(self, model_params):

        self.model.set_params(model_params)

        for i in range(len(model_params)):
            self.latest_model[i] = model_params[i]

    def get_params(self):
        return self.model.get_params()

    def get_grads(self, model_len):
        return self.model.get_gradients(self.train_data, model_len)
    
    def get_raw_grads(self):
        return self.model.get_raw_gradients(self.train_data)

    def set_vzero(self,vzero):
        self.model.set_vzero(vzero)

    def solve_inner(self, num_epochs=1, batch_size=10, debug=False, update=True, local_optim='pgd', term_alpha=0):

        if update:
            bytes_w = self.model.size
            
            local_params = self.model.get_params()
            for i in range(len(self.y_i)):
                self.y_i[i] += self.alpha*(self.latest_model[i] - local_params[i])

            if local_optim != 'gd':
                self.model.optimizer.set_params(self.y_i, self.model)

            soln, comp, iterations = self.model.solve_inner(self.train_data, num_epochs, batch_size, local_optim, term_alpha)
            bytes_r = self.model.size

            self.model.set_params(soln)

            for i in range(len(self.y_i)):
                new_xhat_val = 2*soln[i] - self.y_i[i]
                self.x_hat_diff[i] = new_xhat_val - self.x_hat[i]
                self.x_hat[i] = new_xhat_val

            comp += self.model.size*2

            return (self.num_samples, self.x_hat), (bytes_w, comp, bytes_r), iterations
        else:
            return (self.num_samples, self.x_hat), (0, 0, 0)

    def train_error_and_loss(self):
        tot_correct, loss = self.model.test(self.train_data)
        return tot_correct, loss, self.num_samples


    def test(self):
        tot_correct, loss = self.model.test(self.eval_data)
        return tot_correct, self.test_samples
