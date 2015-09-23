import theano.tensor as T
import numpy as np
import theano

def dropout(rng, values, p):
    mask = rng.binomial(n=1, p=p, size=values.shape, dtype=theano.config.floatX)
    output =  values * mask
    return  np.cast[theano.config.floatX](1.0/p) * output

def sigmoid(x):
	return T.nnet.sigmoid(x)

def tanh(x):
	return T.tanh(x)

def softplus(x):
	return T.log(T.exp(x) + 1)

def relu(x): 
	return x*(x>0)

def relu2(x): 
	return x*(x>0) + 0.01 * x

def initialize_vector(rng, n_out, std=1e-2):
    z = rng.normal(0, std, size=(n_out,))
    return np.asarray(z, dtype=theano.config.floatX)

def initialize_matrix(rng, n_in, n_out):
    z = rng.normal(0, 1, size=(n_in, n_out)) / np.sqrt(n_in)
    return np.asarray(z, dtype=theano.config.floatX)