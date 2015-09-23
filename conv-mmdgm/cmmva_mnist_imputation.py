"""
SVM
"""

import cPickle
import gzip
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
import util.datapy as datapy
from layer import Pegasos
from optimization import optimizer


def svm_cva(dir, predir, start=0, end=500, learning_rate=3e-4, n_epochs=10000,
                           dataset='./data/mnist.pkl.gz',
                           batch_size=500):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """

    ''' 
    Difference
    '''
    print start, end, learning_rate, batch_size

    datasets = datapy.load_data_gpu(dataset, have_matrix=True)

    _, train_set_y, train_y_matrix = datasets[0]
    _, valid_set_y, valid_y_matrix = datasets[1]
    _, test_set_y, test_y_matrix = datasets[2]

    train_set_x, valid_set_x, test_set_x = datapy.load_feature_gpu(dir=dir, start=start,end=end)

    print train_set_x.get_value().shape
    print valid_set_x.get_value().shape
    print test_set_x.get_value().shape

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    '''
    Differences
    '''

    y_matrix = T.imatrix('y_matrix') # labels, presented as 2D matrix of int labels 

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    rng = np.random.RandomState(0)
    n_in=end-start
    classifier = Pegasos.Pegasos(input=x, rng=rng, n_in=n_in, n_out=10,  weight_decay=1e-4, loss=1)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.objective(10, y, y_matrix)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            #y_matrix: test_y_matrix[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            #y_matrix: valid_y_matrix[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    params = [classifier.W, classifier.b]
    grads = [g_W, g_b]

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    
    l_r = theano.shared(np.asarray(learning_rate, dtype=np.float32))
    #get_optimizer = optimizer.get_simple_optimizer(learning_rate=learning_rate)
    get_optimizer = optimizer.get_adam_optimizer_min(learning_rate=l_r, decay1 = 0.1, decay2 = 0.001)
    updates = get_optimizer(params,grads)
    

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=[cost],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            y_matrix: train_y_matrix[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
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

    logdir = dir + str(learning_rate)+'_c-'

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            #print minibatch_avg_cost
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                this_test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                this_test_score = np.mean(this_test_losses)

                if this_test_score < best_test_score:
                    best_test_score = this_test_score

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
                    with open(logdir+'hook.txt', 'a') as f:
                        print >>f,(
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

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    with open(logdir+'hook.txt', 'a') as f:
        print>>f,(
            (
                'Optimization complete with best validation score of %f %%,'
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., test_score * 100.)
        )
        print>>f, 'The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time))
        print>>f, sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.1fs' % ((end_time - start_time)))
        print>>f, best_test_score

    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))
    print best_test_score

    if predir is not None:
        # output put the joint result
        pre_train = np.load(predir+'model-600.npz')
        pre_train = pre_train['model']
        pw = pre_train[-2]
        pb = pre_train[-1]
        params[0].set_value(pw)
        params[1].set_value(pb)
        ptest_losses = [test_model(i) for i in xrange(n_test_batches)]
        ptest_score = np.mean(ptest_losses)
        with open(logdir+'hook.txt', 'a') as f:
            print >>f, 'Jointly trained classifier', ptest_score
        print 'Jointly trained classifier', ptest_score

if __name__ == '__main__':
    learning_rate = float(sys.argv[1])
    dir = sys.argv[3]
    predir = sys.argv[2]
    start=0
    end=500   
    svm_cva(dir=dir, predir=predir, start=start, end=end, learning_rate=learning_rate)