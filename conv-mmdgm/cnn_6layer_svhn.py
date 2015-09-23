"""
simple deep cnn
"""

import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T

from layer import ConvMaxPool_GauInit_DNN
from layer import FullyConnected
from layer import LogisticRegression, Pegasos
from util import datapy, color
from layer import nonlinearity
from optimization import optimizer_separated

def deep_cnn_6layer_svhn_final_svm(learning_rate=0.01,
            n_epochs=500,
            dataset='svhngcn_var',
            batch_size=500,
            dropout_flag=1,
            seed=0,
            predir=None,
            preepoch=10,
            activation=None,
            weight_decay=1e-4):
    
    '''
    svhn
    '''
    n_channels = 3
    dim_w = 32
    dim_h = 32
    n_classes = 10
    
    epoch_threshold = 200
    if os.environ.has_key('epoch_threshold'):
        epoch_threshold = int(os.environ['epoch_threshold'])
    first_drop=0.6
    if os.environ.has_key('first_drop'):
        first_drop = float(os.environ['first_drop'])
    last_drop=1
    if os.environ.has_key('last_drop'):
        last_drop = float(os.environ['last_drop'])
    nkerns_1=96
    if os.environ.has_key('nkerns_1'):
        nkerns_1 = int(os.environ['nkerns_1'])
    nkerns_2=96
    if os.environ.has_key('nkerns_2'):
        nkerns_2 = int(os.environ['nkerns_2'])
    opt_med='adam'
    if os.environ.has_key('opt_med'):
        opt_med = os.environ['opt_med']
    std = 2e-2
    if os.environ.has_key('std'):
        std = os.environ['std']
    pattern = 'hinge'
    if os.environ.has_key('pattern'):
        pattern = os.environ['pattern']
    Loss_L = 1
    if os.environ.has_key('Loss_L'):
        Loss_L = float(os.environ['Loss_L'])

    #cp->cd->cpd->cd->c
    nkerns=[nkerns_1, nkerns_1, nkerns_1, nkerns_2, nkerns_2]
    drops=[0, 1, 1, 1, 0, 1]
    drop_p=[1, first_drop, first_drop, first_drop, 1, last_drop]
    #skerns=[5, 3, 3, 3, 3]
    #pools=[2, 1, 2, 1, 1]
    #modes=['same']*5

    
    logdir = 'results/supervised/cnn/svhn/deep_cnn_6layer_'+pattern+'_'+dataset+str(nkerns)+str(drops)+'_'+str(weight_decay)+'_'+str(learning_rate)+'_'+str(std)+'_'+str(Loss_L)+'_'+str(int(time.time()))+'/'
    if dropout_flag==1:
        logdir = 'results/supervised/cnn/svhn/deep_cnn_6layer_'+pattern+'_'+dataset+str(drop_p)+str(nkerns)+str(drops)+'_'+str(weight_decay)+'_'+str(learning_rate)+'_'+str(std)+'_'+str(Loss_L)+'_dropout_'+str(int(time.time()))+'/'
    if not os.path.exists(logdir): os.makedirs(logdir)
    print 'logdir:', logdir
    print 'deep_cnn_6layer_svm', nkerns, drops, drop_p, seed, dropout_flag
    print 'epoch_threshold', epoch_threshold, 'opt_med', opt_med
    with open(logdir+'hook.txt', 'a') as f:
        print >>f, 'logdir:', logdir
        print >>f, 'epoch_threshold', epoch_threshold, 'opt_med', opt_med
        print >>f, 'deep_cnn_6layer_svm', nkerns, drops, drop_p, seed, dropout_flag

    rng = np.random.RandomState(0)
    rng_share = theano.tensor.shared_randomstreams.RandomStreams(0)

    color.printRed('dataset '+dataset)

    datasets = datapy.load_data_svhn(dataset, have_matrix=True)
    train_set_x, train_set_y, train_y_matrix = datasets[0]
    test_set_x, test_set_y, test_y_matrix = datasets[1]
    valid_set_x, valid_set_y, valid_y_matrix = datasets[2]
    #datasets = datapy.load_data_svhn(dataset, have_matrix=False)
    #train_set_x, train_set_y = datasets[0]
    #test_set_x, test_set_y = datasets[1]
    #valid_set_x, valid_set_y = datasets[2]

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

    layer0_input = x.reshape((batch_size, n_channels, dim_h, dim_w))
    
    if activation =='nonlinearity.relu':
        activation = nonlinearity.relu
    elif activation =='nonlinearity.tanh':
        activation = nonlinearity.tanh
    elif activation =='nonlinearity.softplus':
        activation = nonlinearity.softplus
    
    recg_layer = []
    cnn_output = []
    l = []
    d = []

    #1
    recg_layer.append(ConvMaxPool_GauInit_DNN.ConvMaxPool_GauInit_DNN(
        rng,
        image_shape=(batch_size, n_channels, dim_h, dim_w),
        filter_shape=(nkerns[0], n_channels, 5, 5),
        poolsize=(2, 2),
        border_mode='same', 
        activation=activation,
        std=std
    ))
    if drops[0]==1:
        cnn_output.append(recg_layer[-1].drop_output(layer0_input, drop=drop, rng=rng_share, p=drop_p[0]))
    else:
        cnn_output.append(recg_layer[-1].output(layer0_input))
    l+=[1, 2]
    d+=[1, 0]

    #2
    recg_layer.append(ConvMaxPool_GauInit_DNN.ConvMaxPool_GauInit_DNN(
        rng,
        image_shape=(batch_size, nkerns[0], 16, 16),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(1, 1),
        border_mode='same', 
        activation=activation,
        std=std
    ))
    if drops[1]==1:
        cnn_output.append(recg_layer[-1].drop_output(cnn_output[-1], drop=drop, rng=rng_share, p=drop_p[1]))
    else:
        cnn_output.append(recg_layer[-1].output(cnn_output[-1]))
    l+=[1, 2]
    d+=[1, 0]

    #3
    recg_layer.append(ConvMaxPool_GauInit_DNN.ConvMaxPool_GauInit_DNN(
        rng,
        image_shape=(batch_size, nkerns[1], 16, 16),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=(2, 2),
        border_mode='same', 
        activation=activation,
        std=std
    ))
    if drops[2]==1:
        cnn_output.append(recg_layer[-1].drop_output(cnn_output[-1], drop=drop, rng=rng_share, p=drop_p[2]))
    else:
        cnn_output.append(recg_layer[-1].output(cnn_output[-1]))
    l+=[1, 2]
    d+=[1, 0]

    #4
    recg_layer.append(ConvMaxPool_GauInit_DNN.ConvMaxPool_GauInit_DNN(
        rng,
        image_shape=(batch_size, nkerns[2], 8, 8),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),
        poolsize=(1, 1),
        border_mode='same', 
        activation=activation,
        std=std
    ))
    if drops[3]==1:
        cnn_output.append(recg_layer[-1].drop_output(cnn_output[-1], drop=drop, rng=rng_share, p=drop_p[3]))
    else:
        cnn_output.append(recg_layer[-1].output(cnn_output[-1]))
    
    l+=[1, 2]
    d+=[1, 0]

    #5
    recg_layer.append(ConvMaxPool_GauInit_DNN.ConvMaxPool_GauInit_DNN(
        rng,
        image_shape=(batch_size, nkerns[3], 8, 8),
        filter_shape=(nkerns[4], nkerns[3], 3, 3),
        poolsize=(2, 2),
        border_mode='same', 
        activation=activation,
        std=std
    ))
    if drops[4]==1:
        cnn_output.append(recg_layer[-1].drop_output(cnn_output[-1], drop=drop, rng=rng_share, p=drop_p[4]))
    else:
        cnn_output.append(recg_layer[-1].output(cnn_output[-1]))
    l+=[1, 2]
    d+=[1, 0]

    feature = cnn_output[-1].flatten(2)

    # classify the values of the fully-connected sigmoidal layer
    
    '''
    large weight of pegasos to avoid gradient disappeared 
    '''
    std_pegasos=std
    weight_decay_pegasos=weight_decay
    classifier = Pegasos.Pegasos(input=feature, rng=rng, n_in=nkerns[-1]*4*4, n_out=n_classes, weight_decay=0, loss=Loss_L, std=std_pegasos, pattern=pattern)
    #classifier = LogisticRegression.LogisticRegression(
    #        input=feature,
    #        n_in=nkerns[-1],
    #        n_out=n_classes
    #    )

    l+=[1, 2]
    d+=[weight_decay_pegasos / weight_decay, 0]
    # the cost we minimize during training is the NLL of the model
    cost = classifier.hinge_loss(n_classes, y, y_matrix)
    #cost = classifier.negative_log_likelihood(y)

    params=[]
    for r in recg_layer:
        params+=r.params
    params += classifier.params
    
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    l_r = theano.shared(np.asarray(learning_rate, dtype=np.float32))
    if opt_med=='adam':
        get_optimizer = optimizer_separated.get_adam_optimizer_min(learning_rate=l_r, decay1 = 0.1, decay2 = 0.001, weight_decay=weight_decay)
    elif opt_med=='mom':
        get_optimizer = optimizer_separated.get_momentum_optimizer_min(learning_rate=l_r, weight_decay=weight_decay)
    updates = get_optimizer(w=params,g=grads, l=l, d=d)
    
    pog = []
    for (p,g) in zip(params, grads):
        pog.append(p.max())
        pog.append((p**2).mean())
        pog.append((g**2).mean())
        pog.append((T.sqrt(pog[-2] / pog[-1]))/ 1e3)

    paramovergrad = theano.function(
        inputs=[index],
        outputs=pog,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            y_matrix: train_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](dropout_flag)
        }
    )

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

    valid_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0)
        }
    )

    train_activations = theano.function(
        inputs=[index],
        outputs=feature,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0)
        }
    )
    
    test_activations = theano.function(
        inputs=[index],
        outputs=feature,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0)
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


    if predir is not None:
        color.printBlue('... setting parameters')
        color.printBlue(predir)
        pre_train = np.load(predir+'svhn_model-'+str(preepoch)+'.npz')
        pre_train = pre_train['model']
        for (para, pre) in zip(params, pre_train):
            para.set_value(pre)
        this_test_losses = [test_model(i) for i in xrange(n_test_batches)]
        this_test_score = np.mean(this_test_losses)
        #print predir
        print 'preepoch', preepoch, 'prescore', this_test_score
        with open(logdir+'hook.txt', 'a') as f:
            print >>f, predir
            print >>f, 'preepoch', preepoch, 'prescore', this_test_score

        

    print '... training'
    validation_frequency = n_train_batches/10
    best_train_loss = 10000.0
    best_valid_score = 10000.0
    best_epoch = 0
    test_score = 0
    start_time = time.clock()
    epoch = 0
    n_epochs = 100
    test_epochs = 40
    record = 0
    

    '''
    pog = [paramovergrad(i) for i in xrange(n_train_batches)]
    pog = np.mean(pog, axis=0)
    #print 'before train ----------pog', pog
    with open(logdir+'hook.txt', 'a') as f:
        print >>f, 'before train ----------pog', pog
    '''
    
    while (epoch < n_epochs):
        epoch = epoch + 1
        tmp1 = time.clock()
        preW = None
        currentW = None
        minibatch_avg_cost = 0
        train_error = 0
        if (epoch - record) >= 7:
            record = epoch
            l_r.set_value(np.cast['float32'](l_r.get_value()/3.0))
            print '---------', epoch, l_r.get_value()
            with open(logdir+'hook.txt', 'a') as f:
                print >>f,'---------', epoch, l_r.get_value()
        '''
        decay_epoch = epoch - test_epochs
        if decay_epoch > 0 and decay_epoch % 30==0:
            l_r.set_value(np.cast['float32'](l_r.get_value()/3.0))
            print '---------', epoch, l_r.get_value()
            with open(logdir+'hook.txt', 'a') as f:
                print >>f,'---------', epoch, l_r.get_value()
        '''

        if epoch%5==0:   
            ''' 
            for i in xrange(n_train_batches):
                if i == 0:
                    train_features = np.asarray(train_activations(i))
                else:
                    train_features = np.vstack((train_features, np.asarray(train_activations(i))))
            for i in xrange(n_test_batches):
                if i == 0:
                    test_features = np.asarray(test_activations(i))
                else:
                    test_features = np.vstack((test_features, np.asarray(test_activations(i))))
            
            np.save(logdir+'train_features-'+str(epoch), train_features)
            np.save(logdir+'test_features-'+str(epoch), test_features)
            '''
            model = parameters()
            for i in xrange(len(model)):
                model[i] = np.asarray(model[i]).astype(np.float32)

            np.savez(logdir+'svhn_model-'+str(epoch), model=model)
            
        for minibatch_index in xrange(n_train_batches):
            
            if (minibatch_index <11):
                preW = currentW
                currentW = parameters()
                for i in xrange(len(currentW)):
                    currentW[i] = np.asarray(currentW[i]).astype(np.float32)

                if preW is not None:
                    for (c,p) in zip(currentW, preW):
                        #print minibatch_index, (c**2).mean(), ((c-p)**2).mean(), np.sqrt((c**2).mean()/((c-p)**2).mean())
                        with open(logdir+'delta_w.txt', 'a') as f:
                            print >>f,minibatch_index, (c**2).mean(), ((c-p)**2).mean(), np.sqrt((c**2).mean()/((c-p)**2).mean())
                    
            co, te = train_model(minibatch_index)
            minibatch_avg_cost+=co
            train_error+=te
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                print epoch, minibatch_index
                with open(logdir+'hook.txt', 'a') as f:
                    print >>f, epoch, minibatch_index
                print 'Stochastic hinge loss and training error', minibatch_avg_cost / float(minibatch_index), train_error / float(minibatch_index)
                #print 'time', time.clock() - tmp1
                with open(logdir+'hook.txt', 'a') as f:
                #    print >>f, 'pog', pog
                    print >>f,'Stochastic hinge loss and training error', minibatch_avg_cost / float(minibatch_index), train_error / float(minibatch_index)
                    #print >>f,'time', time.clock() - tmp1

                this_valid_losses = [valid_model(i) for i in xrange(n_valid_batches)]
                this_valid_score = np.mean(this_valid_losses)

                print(
                    'epoch %i, minibatch %i/%i, valid error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        #this_validation_loss * 100,
                        this_valid_score *100.
                    )
                )
                with open(logdir+'hook.txt', 'a') as f:
                    print >>f, (
                        'epoch %i, minibatch %i/%i, valid error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            #this_validation_loss * 100,
                            this_valid_score *100.
                        )
                    )
                if this_valid_score < best_valid_score:
                    this_test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    this_test_score = np.mean(this_test_losses)
                    best_valid_score = this_valid_score
                    test_score = this_test_score
                    best_epoch = epoch
                    record = epoch
                    print 'Update best model', this_test_score
                    with open(logdir+'hook.txt', 'a') as f:
                        print >>f,'Update best model', this_test_score
                print 'So far best model', best_epoch, test_score
                with open(logdir+'hook.txt', 'a') as f:
                    print >>f, 'So far best model', best_epoch, test_score

        pogzero = np.asarray(paramovergrad(0))
        #print 'pogzero', pogzero
        with open(logdir+'pog.txt', 'a') as f:
            print >>f, 'pogzero', pogzero
            
        #pog = [paramovergrad(i) for i in xrange(n_train_batches)]
        #pog = np.mean(pog, axis=0)
        #print 'pog', pog

    print 'So far best model', test_score
    with open(logdir+'hook.txt', 'a') as f:
        print >>f, 'So far best model', test_score
        
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
    weight_decay = float(sys.argv[4])
    dataset = 'svhn'+sys.argv[5]
    if len(sys.argv) > 6:
        predir = sys.argv[6]
        preepoch = int(sys.argv[7])
        deep_cnn_6layer_svhn_final_svm(dataset=dataset,learning_rate=learning_rate,
            activation=activation, dropout_flag=dropout_flag, predir=predir, preepoch=preepoch, weight_decay=weight_decay)
    else:
        deep_cnn_6layer_svhn_final_svm(dataset=dataset,learning_rate=learning_rate,
            activation=activation, dropout_flag=dropout_flag, weight_decay=weight_decay)