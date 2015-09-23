"""
A simple deep cnn on mnist
"""

import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T

from layer import ConvMaxPool
from layer import FullyConnected
from layer import Pegasos
from layer import nonlinearity
from util import datapy
from optimization import optimizer

def deep_cnn_6layer_mnist_50000(learning_rate=3e-4,
            n_epochs=250,
            dataset='mnist.pkl.gz',
            batch_size=500,
            dropout_flag=0,
            seed=0,
            activation=None):
    
    #cp->cd->cpd->cd->c
    nkerns=[32, 32, 64, 64, 64]
    drops=[1, 0, 1, 0, 0]
    #skerns=[5, 3, 3, 3, 3]
    #pools=[2, 1, 1, 2, 1]
    #modes=['same']*5
    n_hidden=[500]

    
    logdir = 'results/supervised/cnn/mnist/deep_cnn_6layer_50000_'+str(nkerns)+str(drops)+str(n_hidden)+'_'+str(learning_rate)+'_'+str(int(time.time()))+'/'
    if dropout_flag==1:
        logdir = 'results/supervised/cnn/mnist/deep_cnn_6layer_50000_'+str(nkerns)+str(drops)+str(n_hidden)+'_'+str(learning_rate)+'_dropout_'+str(int(time.time()))+'/'
    if not os.path.exists(logdir): os.makedirs(logdir)
    print 'logdir:', logdir
    print 'deep_cnn_6layer_mnist_50000_', nkerns, n_hidden, drops, seed, dropout_flag
    with open(logdir+'hook.txt', 'a') as f:
        print >>f, 'logdir:', logdir
        print >>f, 'deep_cnn_6layer_mnist_50000_', nkerns, n_hidden, drops, seed, dropout_flag

    rng = np.random.RandomState(0)
    rng_share = theano.tensor.shared_randomstreams.RandomStreams(0)
    '''
    '''
    datasets = datapy.load_data_gpu_60000(dataset, have_matrix=True)

    train_set_x, train_set_y, train_y_matrix = datasets[0]
    valid_set_x, valid_set_y, valid_y_matrix = datasets[1]
    test_set_x, test_set_y, test_y_matrix = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    '''
    dropout
    '''
    drop = T.iscalar('drop')

    y_matrix = T.imatrix('y_matrix') # labels, presented as 2D matrix of int labels 

    print '... building the model'

    layer0_input = x.reshape((batch_size, 1, 28, 28))
    
    if activation =='nonlinearity.relu':
        activation = nonlinearity.relu
    elif activation =='nonlinearity.tanh':
        activation = nonlinearity.tanh
    elif activation =='nonlinearity.softplus':
        activation = nonlinearity.softplus
    
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
        cnn_output.append(recg_layer[-1].drop_output(layer0_input, drop=drop, rng=rng_share))
    else:
        cnn_output.append(recg_layer[-1].output(layer0_input))

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
            drop: np.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0)
        }
    )

    train_model_average = theano.function(
        inputs=[index],
        outputs=[cost, classifier.errors(y)],
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            y_matrix: train_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](dropout_flag)
        }
    )

    train_model = theano.function(
        inputs=[index],
        outputs=[cost, classifier.errors(y)],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            y_matrix: train_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](dropout_flag)
        }
    )

    print '... training'
    # early-stopping parameters
    patience = n_train_batches * 100  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_test_score = np.inf
    test_score = 0.
    start_time = time.clock()
    epoch = 0
    decay_epochs = 150

    while (epoch < n_epochs):
        epoch = epoch + 1
        tmp1 = time.clock()

        minibatch_avg_cost = 0
        train_error = 0

        for minibatch_index in xrange(n_train_batches):

            co, te = train_model(minibatch_index)
            minibatch_avg_cost+=co
            train_error+=te
            #print minibatch_avg_cost
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                test_epoch = epoch - decay_epochs
                if test_epoch > 0 and test_epoch % 10 == 0:
                    print l_r.get_value()
                    with open(logdir+'hook.txt', 'a') as f:
                        print >>f,l_r.get_value()
                    l_r.set_value(np.cast['float32'](l_r.get_value()/3.0))

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                this_test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                this_test_score = np.mean(this_test_losses)

                train_thing = [train_model_average(i) for i in xrange(n_train_batches)]
                train_thing = np.mean(train_thing, axis=0)
                        
                print epoch, 'hinge loss and training error', train_thing
                with open(logdir+'hook.txt', 'a') as f:
                    print >>f, epoch, 'hinge loss and training error', train_thing

                if this_test_score < best_test_score:
                    best_test_score = this_test_score

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%, test error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100,
                        this_test_score *100.
                    )
                )
                with open(logdir+'hook.txt', 'a') as f:
                    print >>f, (
                        'epoch %i, minibatch %i/%i, validation error %f %%, test error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100,
                            this_test_score *100.
                        )
                    )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )
                    with open(logdir+'hook.txt', 'a') as f:
                        print >>f, (
                            (
                                '     epoch %i, minibatch %i/%i, test error of'
                                ' best model %f %%'
                            ) %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                test_score * 100.
                            )
                        )
        
        if epoch%50==0:
            model = parameters()
            for i in xrange(len(model)):
                model[i] = np.asarray(model[i]).astype(np.float32)
            np.savez(logdir+'model-'+str(epoch), model=model)

        print 'hinge loss and training error', minibatch_avg_cost / float(n_train_batches), train_error / float(n_train_batches)
        print 'time', time.clock() - tmp1
        with open(logdir+'hook.txt', 'a') as f:
            print >>f,'hinge loss and training error', minibatch_avg_cost / float(n_train_batches), train_error / float(n_train_batches)
            print >>f,'time', time.clock() - tmp1

    end_time = time.clock()
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))



if __name__ == '__main__':
    activation = 'nonlinearity.'+sys.argv[1]
    dropout_flag = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    deep_cnn_6layer_mnist_50000(learning_rate=learning_rate,
        activation=activation, dropout_flag=dropout_flag)