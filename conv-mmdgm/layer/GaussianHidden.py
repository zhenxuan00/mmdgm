import theano.tensor as T
import theano.tensor.nnet as Tnn
import numpy as np
import theano
import nonlinearity

class GaussianHidden(object):

    """
    Stochastic layer: Gaussian distribution, linear transpose + random sampling z
    """


    def __init__(self, rng, input, n_in, n_out, W_var=None, b_var=None, W_mean=None, b_mean=None,
                 activation=None):

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

        if W_var is None:
            if activation is None:
                W_values_var= np.asarray(
                    np.zeros((n_in, n_out)),
                    dtype=theano.config.floatX
                )
            elif activation == T.tanh or activation == Tnn.sigmoid:
                W_values_var = np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (n_in + n_out)),
                        high=np.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
                )
                if activation == Tnn.sigmoid:
                    W_values_var *= 4
            else:
                raise Exception('Unknown activation in HiddenLayer.')
            W_var = theano.shared(value=W_values_var, name='W', borrow=True)

        if b_var is None:
            b_values_var = np.zeros((n_out,), dtype=theano.config.floatX)
            b_var = theano.shared(value=b_values_var, name='b', borrow=True)

        if W_mean is None:
            if activation is None:
                W_values_mean= nonlinearity.initialize_matrix(rng, n_in, n_out)
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
            b_values_mean = nonlinearity.initialize_vector(rng, n_out)
            b_mean = theano.shared(value=b_values_mean, name='b', borrow=True)

        self.W_var = W_var
        self.W_mean = W_mean
        self.b_var = b_var
        self.b_mean = b_mean

        # N x d_out
        self.q_logvar = (T.dot(input, self.W_var) + self.b_var).clip(-10,10)
        self.q_mean = T.dot(input, self.W_mean) + self.b_mean
        
        # loglikelihood
        self.logpz = -0.5 * (np.log(2 * np.pi) + (self.q_mean**2 + T.exp(self.q_logvar))).sum(axis=1)
        self.logqz = - 0.5 * (np.log(2 * np.pi) + 1 + self.q_logvar).sum(axis=1)

        # parameters of the model
        self.params = [self.W_var, self.b_var, self.W_mean, self.b_mean]

    def sample_z(self, rng_share):
        '''
        eps1 = rng_share.normal(size=(1000,500), dtype=theano.config.floatX)
        print 'eps1', np.mean(eps1.eval()), np.var(eps1.eval())
        '''
        eps = rng_share.normal(size=self.q_mean.shape, dtype=theano.config.floatX)
        return self.q_mean + T.exp(0.5 * self.q_logvar) * eps
        