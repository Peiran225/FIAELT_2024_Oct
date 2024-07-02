from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import flearn.utils.tf_utils as tf_utils


class SVRG(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, use_locking=False, name="SVRG"):
        super(SVRG, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._lr_t = None
        self._lamb = 0

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
    
    def _create_slots(self, var_list):
        
        self.vzero_placeholder = []
        self.f_w_0_placeholder = []
        for v in var_list:
            self._zeros_slot(v, "vzero", self._name)
            self._zeros_slot(v, "f_w_0", self._name)

            self.vzero_placeholder.append(tf.placeholder(tf.float32))
            self.f_w_0_placeholder .append(tf.placeholder(tf.float32))

        self.assign_vzero_op = []
        self.assign_f_w_0_op = []
        for v, vs, l in zip(var_list, self.vzero_placeholder, self.f_w_0_placeholder):
            vzero = self.get_slot(v, 'vzero')
            f_w_0 = self.get_slot(v, 'f_w_0')

            self.assign_vzero_op.append(vzero.assign(vs))
            self.assign_f_w_0_op.append(f_w_0.assign(l))

        self.trainer_assign_vzero = tf.group(*self.assign_vzero_op)
        self.trainer_assign_f_w_0 = tf.group(*self.assign_f_w_0_op)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        f_w_0 = self.get_slot(var, "f_w_0")
        vzero = self.get_slot(var, "vzero")
        v_n_s = grad - f_w_0 + vzero
        var_update = state_ops.assign_sub(var, lr_t * v_n_s)

        return control_flow_ops.group(*[var_update, ])
    
    def set_vzero(self, cog, client):
        feed_dict = {
           placeholder : value 
              for placeholder, value in zip(self.vzero_placeholder, cog)
        }
        with client.graph.as_default():
            client.sess.run(self.trainer_assign_vzero, feed_dict=feed_dict)

    def set_fwzero(self, cog, client):
        feed_dict = {
           placeholder : value 
              for placeholder, value in zip(self.f_w_0_placeholder, cog)
        }
        with client.graph.as_default():
            client.sess.run(self.trainer_assign_f_w_0, feed_dict=feed_dict)

    def set_params(self, cog, client):
        pass

    def update_lbd(self, cog, client):
        pass
