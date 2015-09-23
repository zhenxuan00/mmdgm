import os
import sys
import time
import math

import numpy as np

import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams
from theano.tensor.signal import downsample

from util import datapy, color, paramgraphics
#from optimization import optimizer
from optimization import optimizer_separated
from layer import FullyConnected, nonlinearity
from layer import GaussianHidden, NoParamsGaussianVisiable,Pegasos
#from layer import ConvMaxPool_GauInit_DNN, UnpoolConvNon_GauInit_DNN
from layer import ConvMaxPool_GauInit_DNN, UnpoolConvNon_GauInit_DNN


def cmmva_6layer_svhn(learning_rate=0.01,
            n_epochs=600,
            dataset='svhngcn_var',
            batch_size=500,
            dropout_flag=1,
            seed=0,
            predir=None,
            activation=None,
            n_batch=625,
            weight_decay=1e-4,
            super_predir=None,
            super_preepoch=None):

    """
    Implementation of convolutional MMVA
    """    
    '''
    svhn
    '''
    n_channels = 3
    colorImg = True
    dim_w = 32
    dim_h = 32
    dim_input=(dim_h, dim_w)
    n_classes = 10

    D = 1.0
    C = 1.0
    if os.environ.has_key('C'):
        C = np.cast['float32'](float((os.environ['C'])))
    if os.environ.has_key('D'):
        D = np.cast['float32'](float((os.environ['D'])))
    color.printRed('D '+str(D)+' C '+str(C))
    
    first_drop=0.5
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
    n_z=512
    if os.environ.has_key('n_z'):
        n_z = int(os.environ['n_z'])
    opt_med='adam'
    if os.environ.has_key('opt_med'):
        opt_med = os.environ['opt_med']
    train_logvar=True
    if os.environ.has_key('train_logvar'):
        train_logvar = bool(int(os.environ['train_logvar']))
    std = 2e-2
    if os.environ.has_key('std'):
        std = os.environ['std']
    Loss_L = 1
    if os.environ.has_key('Loss_L'):
        Loss_L = int(os.environ['Loss_L'])
    pattern = 'hinge'
    if os.environ.has_key('pattern'):
        pattern = os.environ['pattern']


    #cp->cd->cpd->cd->c
    nkerns=[nkerns_1, nkerns_1, nkerns_1, nkerns_2, nkerns_2]
    drops=[0, 1, 1, 1, 0, 1]
    drop_p=[1, first_drop, first_drop, first_drop, 1, last_drop]
    n_hidden=[n_z]
    
    logdir = 'results/supervised/cmmva/svhn/cmmva_6layer_'+dataset+pattern+'_D_'+str(D)+'_C_'+str(C)+'_'#+str(nkerns)+str(n_hidden)+'_'+str(weight_decay)+'_'+str(learning_rate)+'_'
    #if predir is not None:
    #    logdir +='pre_'
    #if dropout_flag == 1:
    #    logdir += ('dropout_'+str(drops)+'_')
    #    logdir += ('drop_p_'+str(drop_p)+'_')
    #logdir += ('trainvar_'+str(train_logvar)+'_')
    #logdir += (opt_med+'_')
    #logdir += (str(Loss_L)+'_')
    #if super_predir is not None:
    #    logdir += (str(super_preepoch)+'_')
    logdir += str(int(time.time()))+'/'

    if not os.path.exists(logdir): os.makedirs(logdir)

    print 'logdir:', logdir, 'predir', predir
    print 'cmmva_6layer_svhn_fix', nkerns, n_hidden, seed, dropout_flag, drops, drop_p
    with open(logdir+'hook.txt', 'a') as f:
        print >>f, 'logdir:', logdir, 'predir', predir
        print >>f, 'cmmva_6layer_svhn_fix', nkerns, n_hidden, seed, dropout_flag, drops, drop_p

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
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    random_z = T.matrix('random_z')
    y_matrix = T.imatrix('y_matrix')
    drop = T.iscalar('drop')
    
    activation = nonlinearity.relu

    rng = np.random.RandomState(seed)
    rng_share = theano.tensor.shared_randomstreams.RandomStreams(0)

    input_x = x.reshape((batch_size, n_channels, dim_h, dim_w))
    
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
        cnn_output.append(recg_layer[-1].drop_output(input=input_x, drop=drop, rng=rng_share, p=drop_p[0]))
    else:
        cnn_output.append(recg_layer[-1].output(input=input_x))
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


    mlp_input_x = cnn_output[-1].flatten(2)

    activations = []
    
    activations.append(mlp_input_x)

    classifier = Pegasos.Pegasos(
            input= activations[-1],
            rng=rng,
            n_in=nkerns[-1]*4*4,
            n_out=n_classes,
            weight_decay=0,
            loss=Loss_L,
            pattern=pattern
        )
    l+=[1, 2]
    d+=[1, 0]


    #stochastic layer
    recg_layer.append(GaussianHidden.GaussianHidden(
            rng=rng,
            input=mlp_input_x,
            n_in=4*4*nkerns[-1],
            n_out=n_hidden[0],
            activation=None
        ))
    l+=[1, 2]
    d+=[1, 0]
    l+=[1, 2]
    d+=[1, 0]

    z = recg_layer[-1].sample_z(rng_share)

    gene_layer = []
    z_output = []
    random_z_output = []

    #1
    gene_layer.append(FullyConnected.FullyConnected(
            rng=rng,
            n_in=n_hidden[-1],
            n_out=4*4*nkerns[-1],
            activation=activation
        ))
    
    z_output.append(gene_layer[-1].output(input=z))
    random_z_output.append(gene_layer[-1].output(input=random_z))
    l+=[1, 2]
    d+=[1, 0]
    
    input_z = z_output[-1].reshape((batch_size, nkerns[-1], 4, 4))
    input_random_z = random_z_output[-1].reshape((n_batch, nkerns[-1], 4, 4))
    
    #1
    gene_layer.append(UnpoolConvNon_GauInit_DNN.UnpoolConvNon_GauInit_DNN(
            rng,
            image_shape=(batch_size, nkerns[-1], 4, 4),
            filter_shape=(nkerns[-2], nkerns[-1], 3, 3),
            poolsize=(2, 2),
            border_mode='same', 
            activation=activation
        ))
    l+=[1, 2]
    d+=[1, 0]
    z_output.append(gene_layer[-1].output(input=input_z))
    random_z_output.append(gene_layer[-1].output_random_generation(input=input_random_z, n_batch=n_batch))
    
    #2
    gene_layer.append(UnpoolConvNon_GauInit_DNN.UnpoolConvNon_GauInit_DNN(
            rng,
            image_shape=(batch_size, nkerns[-2], 8, 8),
            filter_shape=(nkerns[-3], nkerns[-2], 3, 3),
            poolsize=(1, 1),
            border_mode='same', 
            activation=activation
        ))
    l+=[1, 2]
    d+=[1, 0]
    z_output.append(gene_layer[-1].output(input=z_output[-1]))
    random_z_output.append(gene_layer[-1].output_random_generation(input=random_z_output[-1], n_batch=n_batch))

    #3
    gene_layer.append(UnpoolConvNon_GauInit_DNN.UnpoolConvNon_GauInit_DNN(
            rng,
            image_shape=(batch_size, nkerns[-3], 8, 8),
            filter_shape=(nkerns[-4], nkerns[-3], 3, 3),
            poolsize=(2, 2),
            border_mode='same', 
            activation=activation
        ))
    l+=[1, 2]
    d+=[1, 0]
    z_output.append(gene_layer[-1].output(input=z_output[-1]))
    random_z_output.append(gene_layer[-1].output_random_generation(input=random_z_output[-1], n_batch=n_batch))

    
    #4
    gene_layer.append(UnpoolConvNon_GauInit_DNN.UnpoolConvNon_GauInit_DNN(
            rng,
            image_shape=(batch_size, nkerns[-4], 16, 16),
            filter_shape=(nkerns[-5], nkerns[-4], 3, 3),
            poolsize=(1, 1),
            border_mode='same', 
            activation=activation
        ))
    l+=[1, 2]
    d+=[1, 0]
    z_output.append(gene_layer[-1].output(input=z_output[-1]))
    random_z_output.append(gene_layer[-1].output_random_generation(input=random_z_output[-1], n_batch=n_batch))


    #5-1 stochastic layer 
    # for this layer, the activation is None to get a Guassian mean
    gene_layer.append(UnpoolConvNon_GauInit_DNN.UnpoolConvNon_GauInit_DNN(
            rng,
            image_shape=(batch_size, nkerns[-5], 16, 16),
            filter_shape=(n_channels, nkerns[-5], 5, 5),
            poolsize=(2, 2),
            border_mode='same', 
            activation=None
        ))
    l+=[1, 2]
    d+=[1, 0]
    x_mean=gene_layer[-1].output(input=z_output[-1])
    random_x_mean=gene_layer[-1].output_random_generation(input=random_z_output[-1], n_batch=n_batch)


    #5-2 stochastic layer 
    # for this layer, the activation is None to get logvar
    if train_logvar:
        gene_layer.append(UnpoolConvNon_GauInit_DNN.UnpoolConvNon_GauInit_DNN(
                rng,
                image_shape=(batch_size, nkerns[-5], 16, 16),
                filter_shape=(n_channels, nkerns[-5], 5, 5),
                poolsize=(2, 2),
                border_mode='same', 
                activation=None
            ))
        l+=[1, 2]
        d+=[1, 0]
        x_logvar=gene_layer[-1].output(input=z_output[-1])
        random_x_logvar=gene_layer[-1].output_random_generation(input=random_z_output[-1], n_batch=n_batch)
    else:
        x_logvar = theano.shared(np.ones((batch_size, n_channels, dim_h, dim_w), dtype='float32'))
        random_x_logvar = theano.shared(np.ones((n_batch, n_channels, dim_h, dim_w), dtype='float32'))

    gene_layer.append(NoParamsGaussianVisiable.NoParamsGaussianVisiable(
            #rng=rng,
            #mean=z_output[-1],
            #data=input_x,
        ))
    logpx = gene_layer[-1].logpx(mean=x_mean, logvar=x_logvar, data=input_x)
    random_x = gene_layer[-1].sample_x(rng_share=rng_share, mean=random_x_mean, logvar=random_x_logvar)

    #L = (logpx + logpz - logqz).sum()
    lowerbound = (
        (logpx + recg_layer[-1].logpz - recg_layer[-1].logqz).mean()
    )
    hinge_loss = classifier.hinge_loss(10, y, y_matrix)
    
    cost = D * lowerbound - C * hinge_loss

    px = (logpx.mean())
    pz = (recg_layer[-1].logpz.mean())
    qz = (- recg_layer[-1].logqz.mean())

    super_params=[]
    for r in recg_layer[:-1]:
        super_params+=r.params
    super_params+=classifier.params

    params=[]
    for g in gene_layer:
        params+=g.params
    for r in recg_layer:
        params+=r.params
    params+=classifier.params
    grads = [T.grad(cost, param) for param in params]

    l_r = theano.shared(np.asarray(learning_rate, dtype=np.float32))
    #get_optimizer = optimizer.get_adam_optimizer(learning_rate=learning_rate)
    if opt_med=='adam':
        get_optimizer = optimizer_separated.get_adam_optimizer_max(learning_rate=l_r, decay1 = 0.1, decay2 = 0.001, weight_decay=weight_decay)
    elif opt_med=='mom':
        get_optimizer = optimizer_separated.get_momentum_optimizer_max(learning_rate=l_r, weight_decay=weight_decay)
    updates = get_optimizer(w=params,g=grads, l=l, d=d)

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y), lowerbound, hinge_loss, cost],
        #outputs=layer[-1].errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size],
            y_matrix: test_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0)
        }
    )

    valid_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y), lowerbound, hinge_loss, cost],
        #outputs=layer[-1].errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size],
            y_matrix: valid_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0)
        }
    )
    

    valid_error = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        #outputs=layer[-1].errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size],
            #y_matrix: valid_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0)
        }
    )




    '''
    Save parameters and activations
    '''

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

    generation_check = theano.function(
        inputs=[index],
        outputs=[x, x_mean.flatten(2), x_logvar.flatten(2)],
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            #y: train_set_y[index * batch_size: (index + 1) * batch_size],
            #y_matrix: train_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0)
        }
    )

    train_activations = theano.function(
        inputs=[index],
        outputs=T.concatenate(activations, axis=1),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0),
            #y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    valid_activations = theano.function(
        inputs=[index],
        outputs=T.concatenate(activations, axis=1),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0),
            #y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_activations = theano.function(
        inputs=[index],
        outputs=T.concatenate(activations, axis=1),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0),
            #y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`

    debug_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y), lowerbound, px, pz, qz, hinge_loss, cost],
        #updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            y_matrix: train_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](dropout_flag)
        }
    )


    random_generation = theano.function(
        inputs=[random_z],
        outputs=[random_x_mean.flatten(2), random_x.flatten(2)],
        givens={
            #drop: np.cast['int32'](0)
        }
    )

    train_bound_without_dropout = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y), lowerbound, hinge_loss, cost],
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            y_matrix: train_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0)
        }
    )

    train_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y), lowerbound, hinge_loss, cost, px, pz, qz, z],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            y_matrix: train_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](dropout_flag),
        }
    )

    ##################
    # Pretrain MODEL #
    ##################
    if predir is not None:
        color.printBlue('... setting parameters')
        color.printBlue(predir)
        pre_train = np.load(predir+'model.npz')
        pre_train = pre_train['model']
        for (para, pre) in zip(params, pre_train):
            para.set_value(pre)
        tmp =  [debug_model(i) for i in xrange(n_train_batches)]
        tmp = (np.asarray(tmp)).mean(axis=0)
        print '------------------', tmp

    if super_predir is not None:
        color.printBlue('... setting parameters')
        color.printBlue(super_predir)
        pre_train = np.load(super_predir+'svhn_model-'+str(super_preepoch)+'.npz')
        pre_train = pre_train['model']
        for (para, pre) in zip(super_params, pre_train):
            para.set_value(pre)
        this_test_losses = [test_model(i) for i in xrange(n_test_batches)]
        this_test_score = np.mean(this_test_losses, axis=0)
        #print predir
        print 'preepoch', super_preepoch, 'pre_test_score', this_test_score
        with open(logdir+'hook.txt', 'a') as f:
            print >>f, predir
            print >>f, 'preepoch', super_preepoch, 'pre_test_score', this_test_score


    ###############
    # TRAIN MODEL #
    ###############

    print '... training'
    validation_frequency = n_train_batches

    predy_valid_stats = [1, 1, 0]
    start_time = time.clock()
    NaN_count = 0
    epoch = 0
    threshold = 0
    generatition_frequency = 1
    if predir is not None:
        threshold = 0
    color.printRed('threshold, '+str(threshold) + 
        ' generatition_frequency, '+str(generatition_frequency)
        +' validation_frequency, '+str(validation_frequency))
    done_looping = False
    n_epochs = 80
    decay_epochs = 40
    record = 0

    '''
    print 'test initialization...'
    pre_model = parameters()
    for i in xrange(len(pre_model)):
        pre_model[i] = np.asarray(pre_model[i])
        print pre_model[i].shape, np.mean(pre_model[i]), np.var(pre_model[i])
    print 'end test...'
    '''
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        minibatch_avg_cost = 0
        train_error = 0
        train_lowerbound = 0
        train_hinge_loss = 0
        _____z = 0
        pxx = 0
        pzz = 0
        qzz = 0
        preW = None
        currentW = None
        
        tmp_start1 = time.clock()
        if epoch == 30:
            validation_frequency = n_train_batches/5
        if epoch == 50:
            validation_frequency = n_train_batches/10

        if epoch == 30 or epoch == 50 or epoch == 70 or epoch == 90:
            record = epoch
            l_r.set_value(np.cast['float32'](l_r.get_value()/3.0))
            print '---------', epoch, l_r.get_value()
            with open(logdir+'hook.txt', 'a') as f:
                print >>f,'---------', epoch, l_r.get_value()
        '''
        test_epoch = epoch - decay_epochs
        if test_epoch > 0 and test_epoch % 5 == 0:
            l_r.set_value(np.cast['float32'](l_r.get_value()/3.0))
            print '---------------', l_r.get_value()
            with open(logdir+'hook.txt', 'a') as f:
                print >>f, '---------------', l_r.get_value()
        '''

        for minibatch_index in xrange(n_train_batches):            
            e, l, h, ttt, tpx, tpz, tqz, _z = train_model(minibatch_index)
            pxx+=tpx
            pzz+=tpz
            qzz+=tqz
            #_____z += (np.asarray(_z)**2).sum() / (n_hidden[-1] * batch_size)
            train_error += e
            train_lowerbound += l
            train_hinge_loss += h
            minibatch_avg_cost += ttt
            
            '''
            llll = debug_model(minibatch_index)
            with open(logdir+'hook.txt', 'a') as f:
                print >>f,'[]', llll
            '''
            if math.isnan(ttt):
                color.printRed('--------'+str(epoch)+'--------'+str(minibatch_index))
                exit()
            

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            '''
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
            ''' 
            # check valid error only, to speed up
            '''
            if (iter + 1) % validation_frequency != 0 and (iter + 1) %(validation_frequency/10) == 0:
                vt = [valid_error(i) for i in xrange(n_valid_batches)]
                vt = np.mean(vt)
                print 'quick valid error', vt
                with open(logdir+'hook.txt', 'a') as f:
                    print >>f, 'quick valid error', vt
                print 'So far best model', predy_valid_stats
                with open(logdir+'hook.txt', 'a') as f:
                    print >>f, 'So far best model', predy_valid_stats
            '''
            

            if (iter + 1) % validation_frequency == 0:
                print minibatch_index, 'stochastic training error', train_error/float(minibatch_index), train_lowerbound/float(minibatch_index), train_hinge_loss/float(minibatch_index), minibatch_avg_cost /float(minibatch_index), pxx/float(minibatch_index), pzz/float(minibatch_index), qzz/float(minibatch_index)#, 'z_norm', _____z/float(minibatch_index)
                with open(logdir+'hook.txt', 'a') as f:
                    print >>f, minibatch_index, 'stochastic training error', train_error/float(minibatch_index), train_lowerbound/float(minibatch_index), train_hinge_loss/float(minibatch_index), minibatch_avg_cost /float(minibatch_index), pxx/float(minibatch_index), pzz/float(minibatch_index), qzz/float(minibatch_index)#, 'z_norm', _____z/float(minibatch_index)
                
                valid_stats = [valid_model(i) for i in xrange(n_valid_batches)]
                this_valid_stats = np.mean(valid_stats, axis=0)

                print epoch, minibatch_index, 'validation stats', this_valid_stats
                #print tmp
                with open(logdir+'hook.txt', 'a') as f:
                    print >>f, epoch, minibatch_index, 'validation stats', this_valid_stats
                print 'So far best model', predy_valid_stats
                with open(logdir+'hook.txt', 'a') as f:
                    print >>f, 'So far best model', predy_valid_stats

                if this_valid_stats[0] < predy_valid_stats[0]:
                    test_stats = [test_model(i) for i in xrange(n_test_batches)]
                    this_test_stats = np.mean(test_stats, axis=0)
                    predy_valid_stats[0] = this_valid_stats[0]
                    predy_valid_stats[1] = this_test_stats[0]
                    predy_valid_stats[2] = epoch
                    record = epoch
                    print 'Update best model', this_test_stats
                    with open(logdir+'hook.txt', 'a') as f:
                        print >>f,'Update best model', this_test_stats
                    model = parameters()
                    for i in xrange(len(model)):
                        model[i] = np.asarray(model[i]).astype(np.float32)
                        #print model[i].shape, np.mean(model[i]), np.var(model[i])
                    np.savez(logdir+'best-model', model=model)

        genezero = generation_check(0)
        with open(logdir+'gene_check.txt', 'a') as f:
            print >>f, 'epoch-----------------------', epoch
            print >>f, 'x', 'x_mean', 'x_logvar'
        '''
        for i in xrange(len(genezero)):
            genezero[i] = np.asarray(genezero[i])
            with open(logdir+'gene_check.txt', 'a') as f:
                print >>f, genezero[i].max(), genezero[i].min(), genezero[i].mean()
        with open(logdir+'gene_check.txt', 'a') as f:
            print >>f, 'norm', np.sqrt(((genezero[0]- genezero[1])**2).sum())
        '''
        if epoch==1:
            xxx = genezero[0]
            image = paramgraphics.mat_to_img(xxx.T, dim_input, colorImg=colorImg, scale=True)
            image.save(logdir+'data.png', 'PNG')
        if epoch%1==0:
            tail='-'+str(epoch)+'.png'
            xxx_now = genezero[1]
            image = paramgraphics.mat_to_img(xxx_now.T, dim_input, colorImg=colorImg, scale=True)
            image.save(logdir+'data_re'+tail, 'PNG')
        
        if math.isnan(minibatch_avg_cost):
            NaN_count+=1
            color.printRed("NaN detected. Reverting to saved best parameters")
            print '---------------NaN_count:', NaN_count
            with open(logdir+'hook.txt', 'a') as f:
                print >>f, '---------------NaN_count:', NaN_count
            
            tmp =  [debug_model(i) for i in xrange(n_train_batches)]
            tmp = (np.asarray(tmp)).mean(axis=0)
            print '------------------NaN check:', tmp
            with open(logdir+'hook.txt', 'a') as f:
               print >>f, '------------------NaN check:', tmp
               
            model = parameters()
            for i in xrange(len(model)):
                model[i] = np.asarray(model[i]).astype(np.float32)
                print model[i].shape, np.mean(model[i]), np.var(model[i])
                print np.max(model[i]), np.min(model[i])
                print np.all(np.isfinite(model[i])), np.any(np.isnan(model[i]))
                with open(logdir+'hook.txt', 'a') as f:
                    print >>f, model[i].shape, np.mean(model[i]), np.var(model[i])
                    print >>f, np.max(model[i]), np.min(model[i])
                    print >>f, np.all(np.isfinite(model[i])), np.any(np.isnan(model[i]))

            best_before = np.load(logdir+'model.npz')
            best_before = best_before['model']
            for (para, pre) in zip(params, best_before):
                para.set_value(pre)
            tmp =  [debug_model(i) for i in xrange(n_train_batches)]
            tmp = (np.asarray(tmp)).mean(axis=0)
            print '------------------', tmp
            return
            
        if epoch%1==0:    
            model = parameters()
            for i in xrange(len(model)):
                model[i] = np.asarray(model[i]).astype(np.float32)
            np.savez(logdir+'model-'+str(epoch), model=model)
        
        tmp_start4=time.clock()

        if epoch % generatition_frequency == 0:
            tail='-'+str(epoch)+'.png'
            random_z = np.random.standard_normal((n_batch, n_hidden[-1])).astype(np.float32)
            _x_mean, _x = random_generation(random_z)
            #print _x.shape
            #print _x_mean.shape
            image = paramgraphics.mat_to_img(_x.T, dim_input, colorImg=colorImg, scale=True)
            image.save(logdir+'samples'+tail, 'PNG')
            image = paramgraphics.mat_to_img(_x_mean.T, dim_input, colorImg=colorImg, scale=True)
            image.save(logdir+'mean_samples'+tail, 'PNG')
            
        #print 'generation_time', time.clock() - tmp_start4
        #print 'one epoch time', time.clock() - tmp_start1

    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    if NaN_count > 0:
        print '---------------NaN_count:', NaN_count
        with open(logdir+'hook.txt', 'a') as f:
            print >>f, '---------------NaN_count:', NaN_count

if __name__ == '__main__':
    
    predir = None
    if os.environ.has_key('predir'):
        predir = os.environ['predir']
    learning_rate=3e-4
    if os.environ.has_key('learning_rate'):
        learning_rate = float(os.environ['learning_rate'])
    weight_decay=4e-3
    if os.environ.has_key('weight_decay'):
        weight_decay = float(os.environ['weight_decay'])
    dropout_flag = 1
    if os.environ.has_key('dropout_flag'):
        dropout_flag = int(os.environ['dropout_flag'])
    dataset = 'svhngcn_var'
    if os.environ.has_key('dataset'):
        dataset = os.environ['dataset']

    super_predir=None
    super_preepoch=None
    if len(sys.argv) > 2:
        super_predir = sys.argv[1]
        super_preepoch = int(sys.argv[2])

    cmmva_6layer_svhn(predir=predir, dropout_flag=dropout_flag, 
        weight_decay=weight_decay, learning_rate=learning_rate,
        dataset=dataset,super_predir=super_predir,super_preepoch=super_preepoch)