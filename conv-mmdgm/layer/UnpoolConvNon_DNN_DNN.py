import os
import sys
import time
import numpy as np
import theano
import nonlinearity

import theano.tensor as T
from theano.sandbox.cuda import dnn
import theano.tensor.extra_ops as Textra


class UnpoolConvNon_DNN_DNN(object):
    """
    Unpool + conv + nonlinearity
    """

    def __init__(self, rng, filter_shape, image_shape, poolsize=(2, 2), border_mode='same', activation=None, mask=None):
        
        assert image_shape[1] == filter_shape[1]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size

        ###--- Change / to *

        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) *
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # store parameters of this layer
        self.params = [self.W, self.b]
        self.poolsize = list(poolsize)
        self.filter_shape = filter_shape
        self.border_mode = border_mode
        self.activation = activation
        if mask is None:
            window = np.zeros((self.poolsize), dtype=np.float32)
            window[0, 0] = 1
            self.mask = theano.shared(np.tile(window.reshape([1, 1]+self.poolsize), image_shape))
        else:
            self.mask = mask
        self.random_mask = None
        #
        #
        #--- Warning, image_shape stored is (2,2) bigger than that used in mask
        #
        #
        i_s = list(image_shape)
        i_s[2] *= poolsize[0]
        i_s[3] *= poolsize[1]
        self.image_shape = i_s
        '''
        dnn
        '''
        self.border = ((filter_shape[2] - 1)/2, (filter_shape[3] - 1)/2)

    def output(self, input, n_batch=None):
        ###--- Unpool

        if self.poolsize[0] == 1 and self.poolsize[1] == 1:
            unpool_out = input
        else:
            unpool_out = Textra.repeat(Textra.repeat(input, self.poolsize[0], axis = 2), self.poolsize[1], axis = 3) * self.mask

        image_shape = list(self.image_shape)
        if n_batch is not None:
            image_shape[0] = n_batch

        ###--- Unpool + conv
        # convolve input feature maps with filters
        if self.border_mode == 'same':
            conv_out = dnn.dnn_conv(
                img=unpool_out,
                kerns=self.W,
                subsample=(1,1),
                border_mode=self.border,
                #conv_mode='cross'
            )
        else:
            raise Exception('Unknown conv type')  

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        lin_output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )

    def drop_output(self, input, drop=0, rng=None, p=0.5):
        ###--- Unpool

        if self.poolsize[0] == 1 and self.poolsize[1] == 1:
            unpool_out = input
        else:
            unpool_out = Textra.repeat(Textra.repeat(input, self.poolsize[0], axis = 2), self.poolsize[1], axis = 3) * self.mask

        image_shape = list(self.image_shape)
        if n_batch is not None:
            image_shape[0] = n_batch

        if self.border_mode == 'same':
            conv_out = dnn.dnn_conv(
                img=unpool_out,
                kerns=self.W,
                subsample=(1,1),
                border_mode=self.border,
                #conv_mode='cross'
            )
        else:
            raise Exception('Unknown conv type')
        

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        lin_output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        output= (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        droppedOutput = nonlinearity.dropout(rng, output, p)
        return T.switch(T.neq(drop, 0), droppedOutput, output)

    def output_random_generation(self, input, n_batch=144):
        ###--- Unpool

        image_shape = list(self.image_shape)
        image_shape[0] = n_batch
        #print '---', image_shape
        if self.random_mask is None:
            image_shape[2]/=self.poolsize[0]
            image_shape[3]/=self.poolsize[1]
            window = np.zeros((self.poolsize), dtype=np.float32)
            window[0, 0] = 1
            self.random_mask = theano.shared(np.tile(window.reshape([1, 1]+self.poolsize), image_shape))
            image_shape[2]*=self.poolsize[0]
            image_shape[3]*=self.poolsize[1]
        #print '----', image_shape

        if self.poolsize[0] == 1 and self.poolsize[1] == 1:
            unpool_out = input
        else:
            unpool_out = Textra.repeat(Textra.repeat(input, self.poolsize[0], axis = 2), self.poolsize[1], axis = 3) * self.random_mask
        
        ###--- Unpool + conv
        # convolve input feature maps with filters

        if self.border_mode == 'same':
            conv_out = dnn.dnn_conv(
                img=unpool_out,
                kerns=self.W,
                subsample=(1,1),
                border_mode=self.border,
                #conv_mode='cross'
            )
        else:
            raise Exception('Unknown conv type')

        '''
        if self.border_mode == 'valid':
            conv_out = conv.conv2d(
                input=unpool_out,
                filters=self.W,
                filter_shape=self.filter_shape,
                image_shape=image_shape,
                border_mode='valid'
            )
        elif self.border_mode == 'same':
            conv_out = conv.conv2d(
                input=unpool_out,
                filters=self.W,
                filter_shape=self.filter_shape,
                image_shape=image_shape,
                border_mode='full'
            )
            padding_w = theano.shared((self.filter_shape[2] - 1) / 2)
            padding_h = theano.shared((self.filter_shape[3] - 1) / 2)
            conv_out = conv_out[:,:,padding_w:-padding_w,padding_h:-padding_h]
        elif self.border_mode == 'full':
            conv_out = conv.conv2d(
                input=unpool_out,
                filters=self.W,
                filter_shape=self.filter_shape,
                image_shape=image_shape,
                border_mode='full'
            )
        else:
            raise Exception('Unknown conv type')
        '''

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        lin_output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
