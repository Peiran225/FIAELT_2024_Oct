import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class PerturbedGradientDescent(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, mu=0.01, use_locking=False, name="PGD"):
        super(PerturbedGradientDescent, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._mu = mu
       
        self._lr_t = None
        self._mu_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._mu_t = ops.convert_to_tensor(self._mu, name="prox_mu")

    def _create_slots(self, var_list):
        
        self.v_star_placeholder = []
        self.lbd_placeholder = []
        for v in var_list:
            self._zeros_slot(v, "vstar", self._name)
            self._zeros_slot(v, "lbd", self._name)

            self.v_star_placeholder.append(tf.placeholder(tf.float32))
            self.lbd_placeholder .append(tf.placeholder(tf.float32))

        self.assign_vstar_op = []
        self.assign_lbd_op = []
        for v, vs, l in zip(var_list, self.v_star_placeholder, self.lbd_placeholder):
            vstar = self.get_slot(v, 'vstar')
            lbd = self.get_slot(v, 'lbd')

            self.assign_vstar_op.append(vstar.assign(vs))
            self.assign_lbd_op.append(lbd.assign(l))

        self.trainer_assign_vstar = tf.group(*self.assign_vstar_op)
        self.trainer_assign_lbd = tf.group(*self.assign_lbd_op)


    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")
        lbd = self.get_slot(var, "lbd")

        var_update = state_ops.assign_sub(var, lr_t*(grad + mu_t*(var-vstar) + lbd))

        return control_flow_ops.group(*[var_update,])
    
    def _apply_sparse_shared(self, grad, var, indices, scatter_add):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")
        lbd = self.get_slot(var, "lbd")

        v_diff = state_ops.assign(vstar, mu_t * (var - vstar) + lbd, use_locking=self._use_locking)

        with ops.control_dependencies([v_diff]):  
            scaled_grad = scatter_add(vstar, indices, grad)
        var_update = state_ops.assign_sub(var, lr_t * scaled_grad)

        return control_flow_ops.group(*[var_update,])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
        grad.values, var, grad.indices,
        lambda x, i, v: state_ops.scatter_add(x, i, v))

    def set_params(self, cog, client):
        feed_dict = {
           placeholder : value 
              for placeholder, value in zip(self.v_star_placeholder, cog)
        }

        with client.graph.as_default():
            client.sess.run(self.trainer_assign_vstar, feed_dict=feed_dict)

    def update_lbd(self, cog, client):
        feed_dict = {
           placeholder : value 
              for placeholder, value in zip(self.lbd_placeholder, cog)
        }

        with client.graph.as_default():
            client.sess.run(self.trainer_assign_lbd, feed_dict=feed_dict)