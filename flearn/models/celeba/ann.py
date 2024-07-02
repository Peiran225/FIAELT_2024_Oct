import numpy as np
import tensorflow as tf
import random
import math

from flearn.utils.model_utils import batch_data, batch_data_multiple_iters
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad, prox_L2

IMAGE_SIZE = 84

class Model(object):
    def __init__(self, num_classes, optimizer, seed=1, train=True):

        # params
        self.num_classes = num_classes
        self.use_in_train = train

        self.graph = tf.Graph()
        self.optimizer = optimizer
        with self.graph.as_default():
            tf.set_random_seed(123+seed)
            if self.use_in_train:
                self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss, self.pred = self.create_model(optimizer)
            else:
                self.features, self.labels, _, self.grads, self.eval_metric_ops, self.loss, self.pred = self.create_model(optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            all_vars = tf.trainable_variables()
            self.model_assign_op = []
            self.model_placeholder = []
            for variable in all_vars:
                self.model_placeholder.append(tf.placeholder(tf.float32))
            
            for variable, val in zip(all_vars, self.model_placeholder):
                self.model_assign_op.append(variable.assign(val))
            self.trainer_assign_model = tf.group(*self.model_assign_op)

        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
    
    def create_model(self, optimizer):
        features = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3), name='features')
        labels = tf.placeholder(tf.int64, shape=[None,], name='labels')
        out = features
        for _ in range(4):
            out = tf.layers.conv2d(out, 32, 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        logits = tf.layers.dense(out, self.num_classes)

        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        if self.use_in_train:
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        else:
            train_op = None
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        
        return features, labels, train_op, grads, eval_metric_ops, loss, predictions["classes"]

    def set_params(self, model_params=None):
        if model_params is not None:
            feed_dict = {
               placeholder : value 
                  for placeholder, value in zip(self.model_placeholder, model_params)
            }
            with self.graph.as_default():
                self.sess.run(self.trainer_assign_model, feed_dict=feed_dict)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):
        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                feed_dict={self.features: data['x'], self.labels: data['y']})
        
        grads = process_grad(model_grads)

        return num_samples, grads
    
    def get_raw_gradients(self, data):
    
        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                                        feed_dict={self.features: data['x'], self.labels: data['y']})

        return model_grads
    
    def set_vzero(self, vzero):
        self.vzero = vzero
    
    def solve_inner(self, data, num_epochs=1, batch_size=32, local_optim='pgd', term_alpha=0):
        iterations = 0
        if local_optim == 'svrg':
            wzero = self.get_params()
            w1 = wzero - self.optimizer._lr * np.array(self.vzero)
            w1 = prox_L2(np.array(w1), np.array(wzero),self.optimizer._lr, self.optimizer._lamb)
            self.set_params(w1)

        break_epoch = False
        for epoch in range(num_epochs):
            if local_optim == 'svrg':
                num_iters = random.choice([i for i in range(1, 1+math.ceil(len(data['x']) / batch_size))])
            else:
                num_iters = math.ceil(len(data['x']) / batch_size)
            for X, y in batch_data_multiple_iters(data, batch_size, num_iters):
                iterations += 1

                if local_optim == 'svrg':
                    current_weight = self.get_params()
                    self.set_params(wzero)
                    fwzero = self.sess.run(self.grads, feed_dict={self.features: X, self.labels: y})
                    self.optimizer.set_fwzero(fwzero, self)
                    self.set_params(current_weight)

                with self.graph.as_default():
                    self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})

                if epoch > 0 and term_alpha> 0:
                    if local_optim == 'svrg':
                        grad_ = flatten_(fwzero)
                    else:
                        with self.graph.as_default():
                            grad_ = self.sess.run(self.grads, feed_dict={self.features: X, self.labels: y})
                        grad_ = flatten_(grad_)
                    
                    param_cur =  self.get_params()
                    param_cur = flatten_(param_cur)
                    
                    if term_alpha*l2_norm(param_cur-param_old) > l2_norm(grad_):
                        break_epoch = True
                        break
                    
                param_old = self.get_params()     
                param_old = flatten_(param_old)

            if break_epoch:
                break
                
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp, iterations
    
    def test(self, data):
        with self.graph.as_default():
            tot_correct, loss, pred = self.sess.run([self.eval_metric_ops, self.loss, self.pred], 
                feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss
    
    def close(self):
        self.sess.close()

def flatten_(param):
    tem = [param[0].flatten()]
    for i in range(len(param)-1):
        tem.append(param[i+1].flatten())
    return np.concatenate(tem, axis=0)

def l2_norm(array):
    return np.sqrt(np.sum(np.square(array)))