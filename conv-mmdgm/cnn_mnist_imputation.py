import os
import sys
import time
import math

import numpy as np

import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams

from util import datapy, color, paramgraphics
from optimization import optimizer
from layer import FullyConnected, nonlinearity
from layer import GaussianHidden, NoParamsBernoulliVisiable,Pegasos
from layer import ConvMaxPool, UnpoolConvNon

def c_6layer_mnist_imputation(seed=0,
             pertub_type=3, pertub_prob=6, pertub_prob1=14,
             predir=None, n_batch=144,
             dataset='mnist.pkl.gz', batch_size=500):

    """
    Missing data imputation
    """    
    #cp->cd->cpd->cd->c
    nkerns=[32, 32, 64, 64, 64]
    drops=[0, 0, 0, 0, 0, 1]
    #skerns=[5, 3, 3, 3, 3]
    #pools=[2, 1, 1, 2, 1]
    #modes=['same']*5
    n_hidden=[500, 50]
    drop_inverses=[1,]
    # 28->12->12->5->5/5*5*64->500->50->500->5*5*64/5->5->12->12->28
    
    if dataset=='mnist.pkl.gz':
        dim_input=(28, 28)
        colorImg=False
   
    train_set_x, test_set_x, test_set_x_pertub, pertub_label, pertub_number = datapy.load_pertub_data(dirs='data_imputation/', pertub_type=pertub_type, pertub_prob=pertub_prob,pertub_prob1=pertub_prob1)
    
    datasets = datapy.load_data_gpu(dataset, have_matrix=True)

    _, train_set_y, train_y_matrix = datasets[0]
    valid_set_x, valid_set_y, valid_y_matrix = datasets[1]
    _, test_set_y, test_y_matrix = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')
    #x_pertub = T.matrix('x_pertub')  # the data is presented as rasterized images
    #p_label = T.matrix('p_label')

    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    y_matrix = T.imatrix('y_matrix')

    drop = T.iscalar('drop')
    drop_inverse = T.iscalar('drop_inverse')
    
    activation = nonlinearity.relu

    rng = np.random.RandomState(seed)
    rng_share = theano.tensor.shared_randomstreams.RandomStreams(0)

    input_x = x.reshape((batch_size, 1, 28, 28))
    
    recg_layer = []
    cnn_output = []

    #1
    recg_layer.append(ConvMaxPool.ConvMaxPool(
            rng,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2),
            border_mode='valid',
            activation=activation
        ))
    if drops[0]==1:
        cnn_output.append(recg_layer[-1].drop_output(input=input_x, drop=drop, rng=rng_share))
    else:
        cnn_output.append(recg_layer[-1].output(input=input_x))

    #2
    recg_layer.append(ConvMaxPool.ConvMaxPool(
        rng,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(1, 1),
        border_mode='same', 
        activation=activation
    ))
    if drops[1]==1:
        cnn_output.append(recg_layer[-1].drop_output(cnn_output[-1], drop=drop, rng=rng_share))
    else:
        cnn_output.append(recg_layer[-1].output(cnn_output[-1]))
    
    #3
    recg_layer.append(ConvMaxPool.ConvMaxPool(
        rng,
        image_shape=(batch_size, nkerns[1], 12, 12),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=(2, 2),
        border_mode='valid', 
        activation=activation
    ))
    if drops[2]==1:
        cnn_output.append(recg_layer[-1].drop_output(cnn_output[-1], drop=drop, rng=rng_share))
    else:
        cnn_output.append(recg_layer[-1].output(cnn_output[-1]))

    #4
    recg_layer.append(ConvMaxPool.ConvMaxPool(
        rng,
        image_shape=(batch_size, nkerns[2], 5, 5),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),
        poolsize=(1, 1),
        border_mode='same', 
        activation=activation
    ))
    if drops[3]==1:
        cnn_output.append(recg_layer[-1].drop_output(cnn_output[-1], drop=drop, rng=rng_share))
    else:
        cnn_output.append(recg_layer[-1].output(cnn_output[-1]))
    #5
    recg_layer.append(ConvMaxPool.ConvMaxPool(
        rng,
        image_shape=(batch_size, nkerns[3], 5, 5),
        filter_shape=(nkerns[4], nkerns[3], 3, 3),
        poolsize=(1, 1),
        border_mode='same', 
        activation=activation
    ))
    if drops[4]==1:
        cnn_output.append(recg_layer[-1].drop_output(cnn_output[-1], drop=drop, rng=rng_share))
    else:
        cnn_output.append(recg_layer[-1].output(cnn_output[-1]))

    mlp_input = cnn_output[-1].flatten(2)

    recg_layer.append(FullyConnected.FullyConnected(
        rng=rng,
        n_in=nkerns[4] * 5 * 5,
        n_out=500,
        activation=activation
    ))

    feature = recg_layer[-1].drop_output(mlp_input, drop=drop, rng=rng_share)

    # classify the values of the fully-connected sigmoidal layer
    classifier = Pegasos.Pegasos(input=feature, rng=rng, n_in=500, n_out=10, weight_decay=0, loss=1)

    # the cost we minimize during training is the NLL of the model
    cost = classifier.hinge_loss(10, y, y_matrix) * batch_size
    weight_decay=1.0/n_train_batches

    # create a list of all model parameters to be fit by gradient descent
    params=[]
    for r in recg_layer:
        params+=r.params
    params += classifier.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    learning_rate = 3e-4
    l_r = theano.shared(np.asarray(learning_rate, dtype=np.float32))
    get_optimizer = optimizer.get_adam_optimizer_min(learning_rate=l_r, decay1 = 0.1, decay2 = 0.001, weight_decay=weight_decay)
    updates = get_optimizer(params,grads)

    '''
    Save parameters and activations
    '''

    parameters = theano.function(
        inputs=[],
        outputs=params,
    )

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            #y_matrix: test_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0)
        }
    )

    test_pertub_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x_pertub[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            #y_matrix: test_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0)
        }
    )


    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            #y_matrix: valid_y_matrix[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0)
        }
    )

    
    ##################
    # Pretrain MODEL #
    ##################

    model_epoch = 250
    if os.environ.has_key('model_epoch'):
        model_epoch = int(os.environ['model_epoch'])
    if predir is not None:
        color.printBlue('... setting parameters')
        color.printBlue(predir)
        if model_epoch == -1:
            pre_train = np.load(predir+'best-model.npz')
        else:
            pre_train = np.load(predir+'model-'+str(model_epoch)+'.npz')
        pre_train = pre_train['model']
        for (para, pre) in zip(params, pre_train):
            para.set_value(pre)
    else:
        exit()

    ###############
    # TRAIN MODEL #
    ###############
    valid_losses = [validate_model(i) for i in xrange(n_valid_batches)]
    valid_score = np.mean(valid_losses)

    test_losses = [test_model(i) for i in xrange(n_test_batches)]
    test_score = np.mean(test_losses)

    test_losses_pertub = [test_pertub_model(i) for i in xrange(n_test_batches)]
    test_score_pertub = np.mean(test_losses_pertub)

    print valid_score, test_score, test_score_pertub

if __name__ == '__main__':
    
    pertub_type = int(sys.argv[1])
    pertub_prob = float(sys.argv[2])
    pertub_prob1 = float(sys.argv[3])
    predir = sys.argv[4]
    c_6layer_mnist_imputation(pertub_type=pertub_type, pertub_prob=pertub_prob, pertub_prob1=pertub_prob1, predir=predir)