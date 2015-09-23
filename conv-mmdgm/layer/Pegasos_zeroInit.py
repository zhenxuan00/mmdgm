import numpy as np
import pickle, gzip
import theano
import theano.tensor as T

class Pegasos_zeroInit:
    """
    A symbolic implementation of pegasos for multiple classification
    """

    def __init__(self, rng, input, n_in, n_out, weight_decay, loss):

        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.weight_decay = np.float32(weight_decay)
        self.loss = np.float32(loss)

        self.L2_reg = (((self.W ** 2).sum())*self.weight_decay/2.0)
        self.linearfunction = T.dot(input, self.W) + self.b
        self.y_pred = T.argmax(self.linearfunction, axis=1)
        self.params = [self.W, self.b]

    def objective(self, nc, y, y_matrix):
        """
        f = lambda/2 * ||w||^2 + \frac{1}{m} 
        \sum max_y'(loss - <w,y> + <w, y'>)
        """
        l_y = (1-y_matrix)*self.loss
        label_result = T.tile(self.linearfunction[T.arange(y.shape[0]), y], (nc,)).reshape((nc,-1))
        return self.L2_reg + T.mean(T.max(l_y - label_result.T + self.linearfunction , axis = 1))

    def hinge_loss(self, nc, y, y_matrix):
        """
        f = lambda/2 * ||w||^2 + \frac{1}{m} 
        \sum max_y'(loss - <w,y> + <w, y'>)
        """
        l_y = (1-y_matrix)*self.loss
        label_result = T.tile(self.linearfunction[T.arange(y.shape[0]), y], (nc,)).reshape((nc,-1))
        return T.mean(T.max(l_y - label_result.T + self.linearfunction , axis = 1))


    def errors(self, y):
        """
        Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
