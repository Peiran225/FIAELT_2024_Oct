import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.optimizer.svrg import SVRG

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
from flearn.utils.utils import History

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        self.params = params

        tf.compat.v1.reset_default_graph()
        
        if params['local_optim'] == "pgd":
            self.inner_opt = PerturbedGradientDescent(params['learning_rate'], 1.0/params['eta'])
        elif params['local_optim'] == "svrg":
            self.inner_opt = SVRG(params['learning_rate'])
        elif params['local_optim'] == "gd":
            self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])

        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        print('Training using FedAvg with {} workers ---'.format(self.clients_per_round))
        history = History(['ComRound','TrainLoss','GradNorm','TrainAcc','TestAcc','GradDiff','NumBytes', 'iter', 'max_iter', 'min_iter'])

        tqdm.write(' -------------------------------------------------------------------------------------------------------------------')
        tqdm.write('| Com. Round | Train Loss | Grad Norm | Train Acc. | Test Acc. | Grad Diff. | Num Bytes | Iter | Max_iter | Min_iter')
        tqdm.write(' -------------------------------------------------------------------------------------------------------------------')

        total_bytes_r = 0
        total_iterations = [0]
        for i in range(self.num_rounds):
            if i % self.eval_every == 0:
                model_len = process_grad(self.latest_model).size
                global_grads = np.zeros(model_len)
                client_grads = np.zeros(model_len)
                num_samples = []
                local_grads = []

                for c in self.clients:
                    num, client_grad = c.get_grads(model_len)
                    local_grads.append(client_grad)
                    num_samples.append(num)
                    global_grads = np.add(global_grads, client_grad * num)
                global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

                grad_norm = np.sqrt(np.sum(np.square(global_grads)))

                difference = 0
                for idx in range(len(self.clients)):
                    difference += np.sum(np.square(global_grads - local_grads[idx]))
                difference = difference * 1.0 / len(self.clients)

                stats = self.test() 
                stats_train = self.train_error_and_loss()

                train_acc = np.sum(stats_train[3])*1.0/np.sum(stats_train[2])
                test_acc = np.sum(stats[3])*1.0/np.sum(stats[2])
                train_loss = np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])
                tqdm.write('| {:10d} | {:10.5f} | {:9.2e} | {:10.5f} | {:9.5f} | {:10.2e} | {:9.2e} | {} | {} | {} |'.format(i, train_loss, grad_norm, train_acc, test_acc, difference, total_bytes_r,
                            sum(total_iterations), max(total_iterations), min(total_iterations)))
                history.update([i, train_loss, grad_norm, train_acc, test_acc, difference, total_bytes_r, sum(total_iterations), max(total_iterations), min(total_iterations)])

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  
            np.random.seed(i)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)

            csolns = []  
            total_iterations = []

            for idx, c in enumerate(active_clients.tolist()):  
                c.set_params(self.latest_model)

                if self.params['local_optim'] == "svrg":
                    self.inner_opt.set_fwzero(self.latest_model, c.model)
                    grads = c.get_raw_grads()
                    c.set_vzero(grads)
                    self.inner_opt.set_vzero(grads, c.model)

                soln, stats, iterations = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size,
                                            local_optim=self.params['local_optim'], term_alpha=self.params['term_alpha'])

                csolns.append(soln)
                total_iterations.append(iterations)

                total_bytes_r += stats[2]

            self.latest_model = self.aggregate(csolns)
            self.client_model.set_params(self.latest_model)

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)
        client_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []

        for c in self.clients:
            num, client_grad = c.get_grads(model_len)
            local_grads.append(client_grad)
            num_samples.append(num)
            global_grads = np.add(global_grads, client_grad * num)
        global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

        grad_norm = np.sqrt(np.sum(np.square(global_grads)))

        difference = 0
        for idx in range(len(self.clients)):
            difference += np.sum(np.square(global_grads - local_grads[idx]))
        difference = difference * 1.0 / len(self.clients)

        stats = self.test() 
        stats_train = self.train_error_and_loss()

        train_acc = np.sum(stats_train[3])*1.0/np.sum(stats_train[2])
        test_acc = np.sum(stats[3])*1.0/np.sum(stats[2])
        train_loss = np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])
        tqdm.write('| {:10d} | {:10.5f} | {:9.2e} | {:10.5f} | {:9.5f} | {:10.2e} | {:9.2e} | {} | {} | {} |'.format(self.num_rounds, train_loss, grad_norm, train_acc, test_acc, difference, total_bytes_r,
                            sum(total_iterations), max(total_iterations), min(total_iterations)))
        tqdm.write(' -------------------------------------------------------------------------------------------------------------------')
        history.update([self.num_rounds, train_loss, grad_norm, train_acc, test_acc, difference, total_bytes_r, sum(total_iterations), max(total_iterations), min(total_iterations)])
        
        return history.get_history(dataframe=True)