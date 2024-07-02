import numpy as np
import tensorflow as tf

def __num_elems(shape):
    tot_elems = 1
    for s in shape:
        tot_elems *= int(s)
    return tot_elems

def graph_size(graph):
    tot_size = 0
    with graph.as_default():
        vs = tf.trainable_variables()
        for v in vs:
            tot_elems = __num_elems(v.shape)
            dtype_size = int(v.dtype.size)
            var_size = tot_elems * dtype_size
            tot_size += var_size
    return tot_size

def process_sparse_grad(grads):

    indices = grads[0].indices
    values =  grads[0].values
    first_layer_dense = np.zeros((80,8))
    for i in range(indices.shape[0]):
        first_layer_dense[indices[i], :] = values[i, :]

    client_grads = first_layer_dense
    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i]) 


    return client_grads

def process_grad(grads):

    client_grads = grads[0]

    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i]) 


    return client_grads

def cosine_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product * 1.0 / (norm_a * norm_b)  

def prox_L2(vt,w0,lr,lam):
    return (1/lr * vt + lam * w0)/(1/lr + lam)



