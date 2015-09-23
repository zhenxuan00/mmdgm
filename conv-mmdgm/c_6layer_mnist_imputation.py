'''
imputation for cmmva and cva on mnist
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
from optimization import optimizer
from layer import FullyConnected, nonlinearity
from layer import GaussianHidden, NoParamsBernoulliVisiable,Pegasos
from layer import ConvMaxPool, UnpoolConvNon

def c_6layer_mnist_imputation(seed=0, ctype='cva',
             pertub_type=3, pertub_prob=6, pertub_prob1=14, visualization_times=20,
             denoise_times=200, predir=None, n_batch=144,
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

    logdir = 'results/imputation/'+ctype+'/mnist/'+ctype+'_6layer_mnist_'+str(pertub_type)+'_'+str(pertub_prob)+'_'+str(pertub_prob1)+'_'+str(denoise_times)+'_'
    logdir += str(int(time.time()))+'/'

    if not os.path.exists(logdir): os.makedirs(logdir)

    print predir
    with open(logdir+'hook.txt', 'a') as f:
        print >>f, predir
   
    train_set_x, test_set_x, test_set_x_pertub, pertub_label, pertub_number = datapy.load_pertub_data(dirs='data_imputation/', pertub_type=pertub_type, pertub_prob=pertub_prob,pertub_prob1=pertub_prob1)
    
    datasets = datapy.load_data_gpu(dataset, have_matrix=True)

    _, _, _ = datasets[0]
    valid_set_x, _, _ = datasets[1]
    _, _, _ = datasets[2]

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
    x_pertub = T.matrix('x_pertub')  # the data is presented as rasterized images
    p_label = T.matrix('p_label')

    random_z = T.matrix('random_z')

    drop = T.iscalar('drop')
    drop_inverse = T.iscalar('drop_inverse')
    
    activation = nonlinearity.relu

    rng = np.random.RandomState(seed)
    rng_share = theano.tensor.shared_randomstreams.RandomStreams(0)

    input_x = x_pertub.reshape((batch_size, 1, 28, 28))
    
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

    mlp_input_x = cnn_output[-1].flatten(2)

    activations = []

    #1
    recg_layer.append(FullyConnected.FullyConnected(
            rng=rng,
            n_in= 5 * 5 * nkerns[-1],
            n_out=n_hidden[0],
            activation=activation
        ))
    if drops[-1]==1:
        activations.append(recg_layer[-1].drop_output(input=mlp_input_x, drop=drop, rng=rng_share))
    else:
        activations.append(recg_layer[-1].output(input=mlp_input_x))

    #stochastic layer
    recg_layer.append(GaussianHidden.GaussianHidden(
            rng=rng,
            input=activations[-1],
            n_in=n_hidden[0],
            n_out = n_hidden[1],
            activation=None
        ))

    z = recg_layer[-1].sample_z(rng_share)


    gene_layer = []
    z_output = []
    random_z_output = []

    #1
    gene_layer.append(FullyConnected.FullyConnected(
            rng=rng,
            n_in=n_hidden[1],
            n_out = n_hidden[0],
            activation=activation
        ))
    
    z_output.append(gene_layer[-1].output(input=z))
    random_z_output.append(gene_layer[-1].output(input=random_z))

    #2
    gene_layer.append(FullyConnected.FullyConnected(
            rng=rng,
            n_in=n_hidden[0],
            n_out = 5*5*nkerns[-1],
            activation=activation
        ))

    if drop_inverses[0]==1:
        z_output.append(gene_layer[-1].drop_output(input=z_output[-1], drop=drop_inverse, rng=rng_share))
        random_z_output.append(gene_layer[-1].drop_output(input=random_z_output[-1], drop=drop_inverse, rng=rng_share))
    else:
        z_output.append(gene_layer[-1].output(input=z_output[-1]))
        random_z_output.append(gene_layer[-1].output(input=random_z_output[-1]))

    input_z = z_output[-1].reshape((batch_size, nkerns[-1], 5, 5))
    input_random_z = random_z_output[-1].reshape((n_batch, nkerns[-1], 5, 5))

    #1
    gene_layer.append(UnpoolConvNon.UnpoolConvNon(
            rng,
            image_shape=(batch_size, nkerns[-1], 5, 5),
            filter_shape=(nkerns[-2], nkerns[-1], 3, 3),
            poolsize=(1, 1),
            border_mode='same', 
            activation=activation
        ))
    
    z_output.append(gene_layer[-1].output(input=input_z))
    random_z_output.append(gene_layer[-1].output_random_generation(input=input_random_z, n_batch=n_batch))
    
    #2
    gene_layer.append(UnpoolConvNon.UnpoolConvNon(
            rng,
            image_shape=(batch_size, nkerns[-2], 5, 5),
            filter_shape=(nkerns[-3], nkerns[-2], 3, 3),
            poolsize=(2, 2),
            border_mode='full', 
            activation=activation
        ))
    
    z_output.append(gene_layer[-1].output(input=z_output[-1]))
    random_z_output.append(gene_layer[-1].output_random_generation(input=random_z_output[-1], n_batch=n_batch))

    #3
    gene_layer.append(UnpoolConvNon.UnpoolConvNon(
            rng,
            image_shape=(batch_size, nkerns[-3], 12, 12),
            filter_shape=(nkerns[-4], nkerns[-3], 3, 3),
            poolsize=(1, 1),
            border_mode='same', 
            activation=activation
        ))
    
    z_output.append(gene_layer[-1].output(input=z_output[-1]))
    random_z_output.append(gene_layer[-1].output_random_generation(input=random_z_output[-1], n_batch=n_batch))

    #4
    gene_layer.append(UnpoolConvNon.UnpoolConvNon(
            rng,
            image_shape=(batch_size, nkerns[-4], 12, 12),
            filter_shape=(nkerns[-5], nkerns[-4], 3, 3),
            poolsize=(1, 1),
            border_mode='same', 
            activation=activation
        ))
    
    z_output.append(gene_layer[-1].output(input=z_output[-1]))
    random_z_output.append(gene_layer[-1].output_random_generation(input=random_z_output[-1], n_batch=n_batch))

    #5 stochastic layer 
    # for the last layer, the nonliearity should be sigmoid to achieve mean of Bernoulli
    gene_layer.append(UnpoolConvNon.UnpoolConvNon(
            rng,
            image_shape=(batch_size, nkerns[-5], 12, 12),
            filter_shape=(1, nkerns[-5], 5, 5),
            poolsize=(2, 2),
            border_mode='full', 
            activation=nonlinearity.sigmoid
        ))

    z_output.append(gene_layer[-1].output(input=z_output[-1]))
    random_z_output.append(gene_layer[-1].output_random_generation(input=random_z_output[-1], n_batch=n_batch))
   
    gene_layer.append(NoParamsBernoulliVisiable.NoParamsBernoulliVisiable(
            #rng=rng,
            #mean=z_output[-1],
            #data=input_x,
        ))
    logpx = gene_layer[-1].logpx(mean=z_output[-1], data=input_x)


    # 4-D tensor of random generation
    random_x_mean = random_z_output[-1]
    random_x = gene_layer[-1].sample_x(rng_share, random_x_mean)

    x_denoised = z_output[-1].flatten(2)
    x_denoised = p_label*x+(1-p_label)*x_denoised

    mse = ((x - x_denoised)**2).sum() / pertub_number

    params=[]
    for g in gene_layer:
        params+=g.params
    for r in recg_layer:
        params+=r.params

    train_activations = theano.function(
        inputs=[index],
        outputs=T.concatenate(activations, axis=1),
        givens={
            x_pertub: train_set_x[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0)
        }
    )

    valid_activations = theano.function(
        inputs=[index],
        outputs=T.concatenate(activations, axis=1),
        givens={
            x_pertub: valid_set_x[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0)
        }
    )

    test_activations = theano.function(
        inputs=[x_pertub],
        outputs=T.concatenate(activations, axis=1),
        givens={
            drop: np.cast['int32'](0)
        }
    )

    imputation_model = theano.function(
        inputs=[index, x_pertub],
        outputs=[x_denoised, mse],
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            p_label:pertub_label[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0),
            drop_inverse: np.cast['int32'](0)
        }
    )

    ##################
    # Pretrain MODEL #
    ##################

    model_epoch = 600
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
    print '... training'

    epoch = 0
    n_visualization = 100
    output = np.ones((n_visualization, visualization_times+2, 784))
    output[:,0,:] = test_set_x.get_value()[:n_visualization,:]
    output[:,1,:] = test_set_x_pertub.get_value()[:n_visualization,:]
    
    image = paramgraphics.mat_to_img(output[:,0,:].T, dim_input, colorImg=colorImg)
    image.save(logdir+'data.png', 'PNG')
    image = paramgraphics.mat_to_img(output[:,1,:].T, dim_input, colorImg=colorImg)
    image.save(logdir+'data_pertub.png', 'PNG')

    tmp = test_set_x_pertub.get_value()

    while epoch < denoise_times:
        epoch = epoch + 1
        this_mse=0
        for i in xrange(n_test_batches):
            d, m = imputation_model(i, tmp[i * batch_size: (i + 1) * batch_size])
            tmp[i * batch_size: (i + 1) * batch_size] = np.asarray(d)
            this_mse+=m
        if epoch<=visualization_times:
            output[:,epoch+1,:] = tmp[:n_visualization,:]

        print epoch, this_mse
        with open(logdir+'hook.txt', 'a') as f:
                print >>f, epoch, this_mse

        image = paramgraphics.mat_to_img(tmp[:n_visualization,:].T, dim_input, colorImg=colorImg)
        image.save(logdir+'procedure-'+str(epoch)+'.png', 'PNG')
        np.savez(logdir+'procedure-'+str(epoch), tmp=tmp)

    image = paramgraphics.mat_to_img((output.reshape(-1,784)).T, dim_input, colorImg=colorImg, tile_shape=(n_visualization,22))
    image.save(logdir+'output.png', 'PNG')
    np.savez(logdir+'output', output=output)

    # save original train features and denoise test features
    for i in xrange(n_train_batches):
        if i == 0:
            train_features = np.asarray(train_activations(i))
        else:
            train_features = np.vstack((train_features, np.asarray(train_activations(i))))

    for i in xrange(n_valid_batches):
        if i == 0:
            valid_features = np.asarray(valid_activations(i))
        else:
            valid_features = np.vstack((valid_features, np.asarray(valid_activations(i))))

    for i in xrange(n_test_batches):
        if i == 0:
            test_features = np.asarray(test_activations(tmp[i * batch_size: (i + 1) * batch_size]))
        else:
            test_features = np.vstack((test_features, np.asarray(test_activations(tmp[i * batch_size: (i + 1) * batch_size]))))
    
    np.save(logdir+'train_features', train_features)
    np.save(logdir+'valid_features', valid_features)
    np.save(logdir+'test_features', test_features)

if __name__ == '__main__':
    
    ctype = sys.argv[1]
    pertub_type = int(sys.argv[2])
    pertub_prob = float(sys.argv[3])
    pertub_prob1 = float(sys.argv[4])
    denoise_times = int(sys.argv[5])
    predir = sys.argv[6]
    c_6layer_mnist_imputation(ctype=ctype, denoise_times=denoise_times,
     pertub_type=pertub_type, pertub_prob=pertub_prob, pertub_prob1=pertub_prob1, predir=predir)