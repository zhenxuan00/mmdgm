'''
extract features for cva on svhn
'''
import os
import sys
import time
import math

import numpy as np

import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams

from util import datapy, color, paramgraphics
#from optimization import optimizer
from optimization import optimizer_separated
from layer import FullyConnected, nonlinearity
from layer import GaussianHidden, NoParamsGaussianVisiable,Pegasos
#from layer import ConvMaxPool_GauInit_DNN, UnpoolConvNon_GauInit_DNN
from layer import ConvMaxPool_GauInit_DNN, UnpoolConvNon_GauInit_DNN

def c_6layer_svhn_features(learning_rate=0.01,
            n_epochs=600,
            dataset='svhngcn_var',
            batch_size=1000,
            dropout_flag=1,
            seed=0,
            predir=None,
            activation=None,
            n_batch=625,
            weight_decay=1e-4,
            super_predir=None,
            super_preepoch=None):
    """
    Missing data imputation
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
    opt_med='mom'
    if os.environ.has_key('opt_med'):
        opt_med = os.environ['opt_med']
    train_logvar=True
    if os.environ.has_key('train_logvar'):
        train_logvar = bool(int(os.environ['train_logvar']))
    dataset='svhnlcn'
    if os.environ.has_key('dataset'):
        dataset = os.environ['dataset']
    n_z=256
    if os.environ.has_key('n_z'):
        n_z = int(os.environ['n_z'])

    #cp->cd->cpd->cd->c
    nkerns=[nkerns_1, nkerns_1, nkerns_1, nkerns_2, nkerns_2]
    drops=[0, 1, 1, 1, 0, 1]
    drop_p=[1, first_drop, first_drop, first_drop, 1, last_drop]
    n_hidden=[n_z]
    
    logdir = 'results/supervised/cva/svhn_features/cva_6layer_svhn'
    if not os.path.exists(logdir): os.makedirs(logdir)
    print 'logdir:', logdir, 'predir', predir

    color.printRed('dataset '+dataset)

    datasets = datapy.load_data_svhn(dataset, have_matrix=False)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]
    valid_set_x, valid_set_y = datasets[2]

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

    p_label = T.matrix('p_label')

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
        activation=activation
    ))
    if drops[0]==1:
        cnn_output.append(recg_layer[-1].drop_output(input=input_x, drop=drop, rng=rng_share, p=drop_p[0]))
    else:
        cnn_output.append(recg_layer[-1].output(input=input_x))
    l+=[1, 2]
    d+=[1, 1]

    #2
    recg_layer.append(ConvMaxPool_GauInit_DNN.ConvMaxPool_GauInit_DNN(
        rng,
        image_shape=(batch_size, nkerns[0], 16, 16),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(1, 1),
        border_mode='same', 
        activation=activation
    ))
    if drops[1]==1:
        cnn_output.append(recg_layer[-1].drop_output(cnn_output[-1], drop=drop, rng=rng_share, p=drop_p[1]))
    else:
        cnn_output.append(recg_layer[-1].output(cnn_output[-1]))
    l+=[1, 2]
    d+=[1, 1]
    
    #3
    recg_layer.append(ConvMaxPool_GauInit_DNN.ConvMaxPool_GauInit_DNN(
        rng,
        image_shape=(batch_size, nkerns[1], 16, 16),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=(2, 2),
        border_mode='same', 
        activation=activation
    ))
    if drops[2]==1:
        cnn_output.append(recg_layer[-1].drop_output(cnn_output[-1], drop=drop, rng=rng_share, p=drop_p[2]))
    else:
        cnn_output.append(recg_layer[-1].output(cnn_output[-1]))
    l+=[1, 2]
    d+=[1, 1]

    #4
    recg_layer.append(ConvMaxPool_GauInit_DNN.ConvMaxPool_GauInit_DNN(
        rng,
        image_shape=(batch_size, nkerns[2], 8, 8),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),
        poolsize=(1, 1),
        border_mode='same', 
        activation=activation
    ))
    if drops[3]==1:
        cnn_output.append(recg_layer[-1].drop_output(cnn_output[-1], drop=drop, rng=rng_share, p=drop_p[3]))
    else:
        cnn_output.append(recg_layer[-1].output(cnn_output[-1]))
    
    l+=[1, 2]
    d+=[1, 1]

    #5
    '''
    --------------------- (2,2) or (4,4)
    '''
    recg_layer.append(ConvMaxPool_GauInit_DNN.ConvMaxPool_GauInit_DNN(
        rng,
        image_shape=(batch_size, nkerns[3], 8, 8),
        filter_shape=(nkerns[4], nkerns[3], 3, 3),
        poolsize=(2, 2),
        border_mode='same', 
        activation=activation
    ))
    if drops[4]==1:
        cnn_output.append(recg_layer[-1].drop_output(cnn_output[-1], drop=drop, rng=rng_share, p=drop_p[4]))
    else:
        cnn_output.append(recg_layer[-1].output(cnn_output[-1]))
    l+=[1, 2]
    d+=[1, 1]

    mlp_input_x = cnn_output[-1].flatten(2)

    activations = []
    activations.append(mlp_input_x)
    #1
    '''
    ---------------------No MLP
    '''
    '''
    recg_layer.append(FullyConnected.FullyConnected(
            rng=rng,
            n_in= 4 * 4 * nkerns[-1],
            n_out=n_hidden[0],
            activation=activation
        ))
    if drops[-1]==1:
        activations.append(recg_layer[-1].drop_output(input=mlp_input_x, drop=drop, rng=rng_share, p=drop_p[-1]))
    else:
        activations.append(recg_layer[-1].output(input=mlp_input_x))
    '''

    #stochastic layer
    recg_layer.append(GaussianHidden.GaussianHidden(
            rng=rng,
            input=activations[-1],
            n_in=4 * 4 * nkerns[-1],
            n_out=n_hidden[0],
            activation=None
        ))
    l+=[1, 2]
    d+=[1, 1]
    l+=[1, 2]
    d+=[1, 1]

    z = recg_layer[-1].sample_z(rng_share)

    gene_layer = []
    z_output = []
    random_z_output = []

    #1
    gene_layer.append(FullyConnected.FullyConnected(
            rng=rng,
            n_in=n_hidden[0],
            n_out = 4*4*nkerns[-1],
            activation=activation
        ))
    
    z_output.append(gene_layer[-1].output(input=z))
    random_z_output.append(gene_layer[-1].output(input=random_z))
    l+=[1, 2]
    d+=[1, 1]

    #2
    '''
    gene_layer.append(FullyConnected.FullyConnected(
            rng=rng,
            n_in=n_hidden[0],
            n_out = 4*4*nkerns[-1],
            activation=activation
        ))
    if drop_inverses[0]==1:
        z_output.append(gene_layer[-1].drop_output(input=z_output[-1], drop=drop_inverse, rng=rng_share))
        random_z_output.append(gene_layer[-1].drop_output(input=random_z_output[-1], drop=drop_inverse, rng=rng_share))
    else:
        z_output.append(gene_layer[-1].output(input=z_output[-1]))
        random_z_output.append(gene_layer[-1].output(input=random_z_output[-1]))
    '''

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
    d+=[1, 1]
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
    d+=[1, 1]
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
    d+=[1, 1]
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
    d+=[1, 1]
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
    d+=[1, 1]
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
        d+=[1, 1]
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

    params=[]
    for g in gene_layer:
        params+=g.params
    for r in recg_layer:
        params+=r.params

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


    ##################
    # Pretrain MODEL #
    ##################
    model_epoch = 100
    ctype = 'cva'
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
        if ctype == 'cva':
            for (para, pre) in zip(params, pre_train):
                para.set_value(pre)
        elif ctype == 'cmmva':
            for (para, pre) in zip(params, pre_train[:-2]):
                para.set_value(pre)
        else:
            exit()
    else:
        exit()

    ###############
    # TRAIN MODEL #
    ###############
    print 'extract features: valid'
    for i in xrange(n_valid_batches):
        if i == 0:
            valid_features = np.asarray(valid_activations(i))
        else:
            valid_features = np.vstack((valid_features, np.asarray(valid_activations(i))))
    #print 'valid'
    print 'extract features: test'
    for i in xrange(n_test_batches):
        if i == 0:
            test_features = np.asarray(test_activations(i))
        else:
            test_features = np.vstack((test_features, np.asarray(test_activations(i))))
    
    f = file(logdir+"svhn_features.bin","wb")
    np.save(f,valid_features)
    np.save(f,test_features)
    f.close()
    #print 'test'

    print 'extract features: train'
    f = file(logdir+"svhn_train_features.bin","wb")
    for i in xrange(n_train_batches):
        #print n_train_batches
        #print i
        train_features = np.asarray(train_activations(i))
        np.save(f,train_features) 
    f.close()

if __name__ == '__main__':
    predir = sys.argv[1]
    c_6layer_svhn_features(predir=predir)