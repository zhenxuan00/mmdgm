import os
import sys
import time
import numpy
import theano
import nonlinearity

import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import theano.tensor.extra_ops as Textra


class UnpoolNonConv(object):
    """
    Unpool + nonlinearity + conv
    """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), border_mode='same', activation=None, mask=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size

        ###--- Change / to *

        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) *
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        ###--- Unpool

        if poolsize[0] == 1 and poolsize[1] == 1:
            self.unpool_out = input
        else:
            if mask is None:
                window = np.zeros((poolsize), dtype=np.float32)
                window[0, 0] = 1
                mask = theano.shared(np.tile(window.reshape([1, 1]+poolsize), input_shape))

            self.unpool_out = Textra.repeat(Textra.repeat(input, poolsize[0], axis = 2), poolsize[1], axis = 3) * mask

        relu_output = (
            self.unpool_out if activation is None
            else activation(self.unpool_out)
        )

        ###--- Unpool + conv
        # convolve input feature maps with filters
        if border_mode == 'valid':
            conv_out = conv.conv2d(
                input=relu_output,
                filters=self.W,
                filter_shape=filter_shape,
                image_shape=image_shape,
                border_mode='valid'
            )
        elif border_mode == 'same':
            conv_out = conv.conv2d(
                input=relu_output,
                filters=self.W,
                filter_shape=filter_shape,
                image_shape=image_shape,
                border_mode='full'
            )
            padding_w = theano.shared((filter_shape[2] - 1) / 2)
            padding_h = theano.shared((filter_shape[3] - 1) / 2)
            conv_out = conv_out[:,:,padding_w:-padding_w,padding_h:-padding_h]
        elif border_mode == 'full':
            conv_out = conv.conv2d(
                input=relu_output,
                filters=self.W,
                filter_shape=filter_shape,
                image_shape=image_shape,
                border_mode='full'
            )
        else:
            raise Exception('Unknown conv type')

        # downsample each feature map individually, using maxpooling
        
        

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output =  conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        

        # store parameters of this layer
        self.params = [self.W, self.b]