import os
import sys
import time
import numpy
import theano
import nonlinearity

import theano.tensor as T
from theano.tensor.signal import downsample
from theano.sandbox.cuda import dnn
import theano.tensor.nnet as Tnn
import ContrastCrossChannels

class ConvMaxPool_GauInit_DNN(object):
    """
    Pool Layer of a convolutional network
    """

    def __init__(self, rng, filter_shape, image_shape, poolsize=(2, 2), border_mode='same', activation=None, std=2e-2, cnorm=False):

        assert image_shape[1] == filter_shape[1]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))

        if activation == nonlinearity.tanh or activation == Tnn.sigmoid:
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        elif activation == nonlinearity.softplus or activation == nonlinearity.relu:
            self.W = theano.shared(
                numpy.asarray(
                    rng.normal(loc=0, scale=std, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            raise Exception('Unknown activation in ConvMaxPool layer.')

        

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # store parameters of this layer
        self.params = [self.W, self.b]
        self.poolsize = poolsize
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.border_mode = border_mode
        '''
        dnn
        '''
        self.border = ((filter_shape[2] - 1)/2, (filter_shape[3] - 1)/2)
        self.activation = activation
        self.cnorm = cnorm

    def output(self, input):
        '''
        dnn
        '''
        if self.border_mode == 'same':
            conv_out = dnn.dnn_conv(
                img=input,
                kerns=self.W,
                subsample=(1,1),
                border_mode=self.border,
                #conv_mode='cross'
            )
        else:
            raise Exception('Unknown conv type')

        # downsample each feature map individually, using maxpooling
        
        if self.poolsize[0] == 1 and self.poolsize[1] == 1:
            pooled_out = conv_out
        else:
            pooled_out = downsample.max_pool_2d(
                input=conv_out,
                ds=self.poolsize,
                ignore_border=True
            )
        if self.cnorm:
            print 'cnorm size', self.filter_shape[0]/8+1
            pooled_out=ContrastCrossChannels.ContrastCrossChannels(input=pooled_out, n=self.filter_shape[0]/8+1)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )

    def drop_output(self, input, drop=0, rng=None, p=0.5):
        # convolve input feature maps with filters
        '''
        dnn
        '''
        if self.border_mode == 'same':
            conv_out = dnn.dnn_conv(
                img=input,
                kerns=self.W,
                subsample=(1,1),
                border_mode=self.border,
                #conv_mode='cross'
            )
        else:
            raise Exception('Unknown conv type')

        # downsample each feature map individually, using maxpooling
        
        if self.poolsize[0] == 1 and self.poolsize[1] == 1:
            pooled_out = conv_out
        else:
            pooled_out = downsample.max_pool_2d(
                input=conv_out,
                ds=self.poolsize,
                ignore_border=True
            )
        
        if self.cnorm:
            print 'cnorm size', self.filter_shape[0]/8+1
            pooled_out=ContrastCrossChannels.ContrastCrossChannels(input=pooled_out, n=self.filter_shape[0]/8+1)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        droppedOutput = nonlinearity.dropout(rng, output, p)
        return T.switch(T.neq(drop, 0), droppedOutput, output)