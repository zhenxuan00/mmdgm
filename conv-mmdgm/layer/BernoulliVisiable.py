import theano.tensor as T
import theano.tensor.nnet as Tnn
import numpy as np
import theano
import nonlinearity

class BernoulliVisiable(object):

    """
    Stochastic layer: Bernoulli distribution
    """


    def __init__(self, rng, input, data, n_in, n_out, W_mean=None, b_mean=None, activation=None):

        """
        :rng
        :sampling np for initialization

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the stochastic layer typically
        """
        
        self.input = input

        if W_mean is None:
            if activation is None:
                W_values_mean = nonlinearity.initialize_matrix(rng, n_in, n_out)
            elif activation == T.tanh or activation == Tnn.sigmoid:
                W_values_mean = np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (n_in + n_out)),
                        high=np.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
                )
                if activation == Tnn.sigmoid:
                    W_values_mean *= 4
            else:
                raise Exception('Unknown activation in HiddenLayer.')
            W_mean = theano.shared(value=W_values_mean, name='W', borrow=True)

        if b_mean is None:
            b_values_mean = np.zeros((n_out,), dtype=theano.config.floatX)
            b_mean = theano.shared(value=b_values_mean, name='b', borrow=True)

        self.W_mean = W_mean
        self.b_mean = b_mean

        self.q_mean = T.nnet.sigmoid(T.dot(input, self.W_mean) + self.b_mean)
        
        # loglikelihood
        #self.logpx = - ((self.q_mean - data)**2).sum(axis = 1)
        self.logpx = (- T.nnet.binary_crossentropy(self.q_mean, data)).sum(axis=1)
        
        # parameters of the model
        self.params = [self.W_mean, self.b_mean]

    def sample_x(self, rng_share):
        return rng_share.binomial(n=1,p=self.q_mean,dtype=theano.config.floatX)
        