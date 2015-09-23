import theano.tensor as T
import theano.tensor.nnet as Tnn
import numpy as np
import theano
import nonlinearity
import logpdfs

class NoParamsGaussianVisiable(object):

    """
    Stochastic layer: Unconvolutional gaussian distribution, likelihood + random sampling x
    """
    def __init__(self):
        self.params=[]
    
    def logpx(self, mean, logvar, data):
        _logpx = logpdfs.normal2(data.flatten(2), mean.flatten(2), logvar.flatten(2))
        return _logpx.sum(axis=1)

    def sample_x(self, rng_share, mean, logvar):
        eps = rng_share.normal(size=mean.shape, dtype=theano.config.floatX)
        return mean + T.exp(0.5 * logvar) * eps