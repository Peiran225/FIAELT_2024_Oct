import numpy as np
import tensorflow as tf
from tqdm import tqdm

from flearn.models.client import Client
from flearn.utils.tf_utils import process_grad

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):

        for key, val in params.items(): setattr(self, key, val);

        users, _, _, _ = dataset
        num_clients = len(users)
        if self.clients_per_round < 1: 
            self.clients_per_round = num_clients
        
        self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)
        self.clients = self.setup_clients(dataset, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))

        self.latest_model = self.client_model.get_params()

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, model=None):

        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses
 
    def test(self):
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self):
        pass

    def select_clients(self, round, num_clients=20):
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)  
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return indices, np.asarray(self.clients)[indices]

    def aggregate(self, wsolns):
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])
        for (w, soln) in wsolns:  
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w*v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln

