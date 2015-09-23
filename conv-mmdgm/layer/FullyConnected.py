import theano.tensor as T
import theano.tensor.nnet as Tnn
import numpy
import theano
import nonlinearity

class FullyConnected(object):

    """
    Typical hidden layer of a MLP: units are fully-connected and have
    sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
    and the bias vector b is of shape (n_out,).
    """


    def __init__(self, rng, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        #
        # the output of uniform if converted using asarray to dtype
        #
        # theano.config.floatX so that the code is runable on GPU
        #
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            if activation == T.tanh or activation == nonlinearity.tanh or activation == Tnn.sigmoid:
                W_values = numpy.asarray(
                    rng.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
                )
                if activation == Tnn.sigmoid:
                    W_values *= 4
            elif activation == nonlinearity.softplus or activation == nonlinearity.relu or activation == None:
                W_values = nonlinearity.initialize_matrix(rng, n_in, n_out)
            else:
                raise Exception('Unknown activation in HiddenLayer.')

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = nonlinearity.initialize_vector(rng, n_out)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        # parameters of the model
        self.params = [self.W, self.b]
        self.activation = activation

    def output(self, input):
        lin_output = T.dot(input, self.W) + self.b
        return (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )

    def drop_output(self, input, drop=0, rng=None, p=0.5):
        lin_output = T.dot(input, self.W) + self.b
        output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        droppedOutput = nonlinearity.dropout(rng, output, p)
        return T.switch(T.neq(drop, 0), droppedOutput, output)