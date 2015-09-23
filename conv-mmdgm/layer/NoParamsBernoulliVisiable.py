import theano.tensor as T
import theano.tensor.nnet as Tnn
import numpy as np
import theano
import nonlinearity

class NoParamsBernoulliVisiable(object):

    """
    Stochastic layer: Unconvolutional Bernoulli distribution
    """

    def __init__(self):
        self.params = []

    def logpx(self, mean, data):
        # clip to avoid NaN
        m = mean.clip(0.001,0.999)
        return (- T.nnet.binary_crossentropy(m.flatten(2), data.flatten(2))).sum(axis=1)
    def sample_x(self, rng_share, mean):
        return rng_share.binomial(n=1,p=mean,dtype=theano.config.floatX)
        