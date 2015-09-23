import copy
import logging
import time
import warnings
import os
import numpy
from theano.compat.six.moves import xrange
import scipy
try:
    from scipy import linalg
except ImportError:
    warnings.warn("Could not import scipy.linalg")
import theano
from theano import function, tensor

from pylearn2.blocks import Block
from pylearn2.linear.conv2d import Conv2D
from pylearn2.space import Conv2DSpace, VectorSpace
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils.insert_along_axis import insert_columns
from pylearn2.utils import sharedX
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import contains_nan

"""
    img_shape : WRITEME
    kernel_size : int, optional
        local contrast kernel size
    batch_size: int, optional
        If dataset is based on PyTables use a batch size smaller than
        10000. Otherwise any batch size diffrent than datasize is not
        supported yet.
    threshold : float, Threshold for denominator
    channels : If none, will apply it on all channels.
"""
def gaussian_filter(kernel_shape):
    """
    kernel_shape : WRITEME
    """
    x = numpy.zeros((kernel_shape, kernel_shape),
                    dtype=theano.config.floatX)

    def gauss(x, y, sigma=2.0):
        Z = 2 * numpy.pi * sigma ** 2
        return 1. / Z * numpy.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = numpy.floor(kernel_shape / 2.)
    for i in xrange(0, kernel_shape):
        for j in xrange(0, kernel_shape):
            x[i, j] = gauss(i - mid, j - mid)

    return x / numpy.sum(x)

def lecun_lcn(input, img_shape, kernel_shape, threshold=1e-4):
    """
    Yann LeCun's local contrast normalization

    Original code in Theano by: Guillaume Desjardins

    Parameters
    ----------
    input : WRITEME
    img_shape : WRITEME
    kernel_shape : WRITEME
    threshold : WRITEME
    """
    input = input.reshape((input.shape[0], input.shape[1], input.shape[2], 1))
    X = tensor.matrix(dtype=input.dtype)
    X = X.reshape((len(input), img_shape[0], img_shape[1], 1))

    filter_shape = (1, 1, kernel_shape, kernel_shape)
    filters = sharedX(gaussian_filter(kernel_shape).reshape(filter_shape))

    input_space = Conv2DSpace(shape=img_shape, num_channels=1)
    transformer = Conv2D(filters=filters, batch_size=len(input),
                         input_space=input_space,
                         border_mode='full')
    convout = transformer.lmul(X)

    # For each pixel, remove mean of 9x9 neighborhood
    mid = int(numpy.floor(kernel_shape / 2.))
    centered_X = X - convout[:, mid:-mid, mid:-mid, :]

    # Scale down norm of 9x9 patch if norm is bigger than 1
    transformer = Conv2D(filters=filters,
                         batch_size=len(input),
                         input_space=input_space,
                         border_mode='full')
    sum_sqr_XX = transformer.lmul(X ** 2)

    denom = tensor.sqrt(sum_sqr_XX[:, mid:-mid, mid:-mid, :])
    per_img_mean = denom.mean(axis=[1, 2])
    divisor = tensor.largest(per_img_mean.dimshuffle(0, 'x', 'x', 1), denom)
    divisor = tensor.maximum(divisor, threshold)

    new_X = centered_X / divisor
    new_X = tensor.flatten(new_X, outdim=3)

    f = function([X], new_X)
    return f(input)

def transform(x, channels, img_shape, kernel_size=7, threshold=1e-4):
    """
    ----------
    X : WRITEME
        data with axis [b, 0, 1, c]
    """
    for i in channels:
        assert isinstance(i, int)
        assert i >= 0 and i <= x.shape[3]

        x[:, :, :, i] = lecun_lcn(x[:, :, :, i],
                                  img_shape,
                                  kernel_size,
                                  threshold)
    return x