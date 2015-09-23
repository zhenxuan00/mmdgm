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

def cmmva_6layer_dropout_mnist_60000(seed=0, start_layer=0, end_layer=1, dropout_flag=1, drop_inverses_flag=0, learning_rate=3e-5, predir=None, n_batch=144,
             dataset='mnist.pkl.gz', batch_size=500, nkerns=[20, 50], n_hidden=[500, 50]):

    """
    Implementation of convolutional MMVA
    """    
    #cp->cd->cpd->cd->c
    nkerns=[32, 32, 64, 64, 64]
    drops=[1, 0, 1, 0, 0, 1]
    #skerns=[5, 3, 3, 3, 3]
    #pools=[2, 1, 1, 2, 1]
    #modes=['same']*5
    n_hidden=[500, 50]
    drop_inverses=[1,]
    # 28->12->12->5->5/5*5*64->500->50->500->5*5*64/5->5->12->12->28
    
    if dataset=='mnist.pkl.gz':
        dim_input=(28, 28)
        colorImg=False
    D = 1.0
    C = 1.0
    if os.environ.has_key('C'):
        C = np.cast['float32'](float((os.environ['C'])))
    if os.environ.has_key('D'):
        D = np.cast['float32'](float((os.environ['D'])))
    color.printRed('D '+str(D)+' C '+str(C))

    logdir = 'results/supervised/cmmva/mnist/cmmva_6layer_60000_'+str(nkerns)+str(n_hidden)+'_D_'+str(D)+'_C_'+str(C)+'_'+str(learning_rate)+'_'
    if predir is not None:
        logdir +='pre_'
    if dropout_flag == 1:
        logdir += ('dropout_'+str(drops)+'_')
    if drop_inverses_flag==1:
        logdir += ('inversedropout_'+str(drop_inverses)+'_')
    logdir += str(int(time.time()))+'/'

    if not os.path.exists(logdir): os.makedirs(logdir)
    print 'logdir:', logdir, 'predir', predir
    print 'cmmva_6layer_mnist_60000', nkerns, n_hidden, seed, drops, drop_inverses, dropout_flag, drop_inverses_flag
    with open(logdir+'hook.txt', 'a') as f:
        print >>f, 'logdir:', logdir, 'predir', predir
        print >>f, 'cmmva_6layer_mnist_60000', nkerns, n_hidden, seed, drops, drop_inverses, dropout_flag, drop_inverses_flag

    datasets = datapy.load_data_gpu_60000(dataset, have_matrix=True)

    train_set_x, train_set_y, train_y_matrix = datasets[0]
    valid_set_x, valid_set_y, valid_y_matrix = datasets[1]
    test_set_x, test_set_y, test_y_matrix = datasets[2]

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
    y_matrix = T.imatrix('y_matrix')
    random_z = T.matrix('random_z')

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
    
    features = T.concatenate(activations[start_layer:end_layer], axis=1)
    color.printRed('feature dimension: '+str(np.sum(n_hidden[start_layer:end_layer])))
    
    classifier = Pegasos.Pegasos(
            input= features,
            rng=rng,
            n_in=np.sum(n_hidden[start_layer:end_layer]),
            n_out=10,
            weight_decay=0,
            loss=1,
            std=1e-2
        )

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

    #L = (logpx + logpz - logqz).sum()
    lowerbound = (
        (logpx + recg_layer[-1].logpz - recg_layer[-1].logqz).sum()
    )

    hinge_loss = classifier.hinge_loss(10, y, y_matrix) * batch_size

    #
    # D is redundent, you could just set D = 1 and tune C and weight decay parameters
    # beacuse AdaM is scale-invariant
    #
    cost = D * lowerbound - C * hinge_loss #- classifier.L2_reg
    
    px = (logpx.sum())
    pz = (recg_layer[-1].logpz.sum())
    qz = (- recg_layer[-1].logqz.sum())

    params=[]
    for g in gene_layer:
        params+=g.params
    for r in recg_layer:
        params+=r.params
    params+=classifier.params
    gparams = [T.grad(cost, param) for param in params]

    weight_decay=1.0/n_train_batches
    epsilon=1e-8
    
    #get_optimizer = optimizer.get_adam_optimizer(learning_rate=learning_rate)
    l_r = theano.shared(np.asarray(learning_rate, dtype=np.float32))
    get_optimizer = optimizer.get_adam_optimizer_max(learning_rate=l_r, 
        decay1=0.1, decay2=0.001, weight_decay=weight_decay, epsilon=epsilon)
    with open(logdir+'hook.txt', 'a') as f:
        print >>f, 'AdaM', learning_rate, weight_decay, epsilon
    updates = get_optimizer(params,gparams)

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
            drop: np.cast['int32'](0),
            drop_inverse: np.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y), lowerbound, hinge_loss, cost],
        #outputs=layer[-1].errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size],
            y_matrix: valid_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0),
            drop_inverse: np.cast['int32'](0)
        }
    )

    
    '''
    Save parameters and activations
    '''

    parameters = theano.function(
        inputs=[],
        outputs=params,
    )

    train_activations = theano.function(
        inputs=[index],
        outputs=T.concatenate(activations, axis=1),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0),
            #drop_inverse: np.cast['int32'](0)
            #y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    valid_activations = theano.function(
        inputs=[index],
        outputs=T.concatenate(activations, axis=1),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0),
            #drop_inverse: np.cast['int32'](0)
            #y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_activations = theano.function(
        inputs=[index],
        outputs=T.concatenate(activations, axis=1),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0),
            #drop_inverse: np.cast['int32'](0)
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
            drop: np.cast['int32'](dropout_flag),
            drop_inverse: np.cast['int32'](drop_inverses_flag)
        }
    )

    random_generation = theano.function(
        inputs=[random_z],
        outputs=[random_x_mean.flatten(2), random_x.flatten(2)],
        givens={
            #drop: np.cast['int32'](0),
            drop_inverse: np.cast['int32'](0)
        }
    )
    
    train_bound_without_dropout = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y), lowerbound, hinge_loss, cost],
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            y_matrix: train_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](0),
            drop_inverse: np.cast['int32'](0)
        }
    )

    train_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y), lowerbound, hinge_loss, cost],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            y_matrix: train_y_matrix[index * batch_size: (index + 1) * batch_size],
            drop: np.cast['int32'](dropout_flag),
            drop_inverse: np.cast['int32'](drop_inverses_flag)
        }
    )
    # end-snippet-5

    ##################
    # Pretrain MODEL #
    ##################
    if predir is not None:
        color.printBlue('... setting parameters')
        color.printBlue(predir)
        pre_train = np.load(predir+'model.npz')
        pre_train = pre_train['model']
        # params include w and b, exclude it
        for (para, pre) in zip(params[:-2], pre_train):
            #print pre.shape
            para.set_value(pre)
        tmp =  [debug_model(i) for i in xrange(n_train_batches)]
        tmp = (np.asarray(tmp)).mean(axis=0) / float(batch_size)
        print '------------------', tmp[1:5]
    
    # valid_error test_error  epochs
    predy_test_stats = [1, 1, 0]
    predy_valid_stats = [1, 1, 0]

    best_validation_bound = -1000000.0
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    NaN_count = 0
    epoch = 0
    threshold = 0
    validation_frequency = 1
    generatition_frequency = 10
    if predir is not None:
        threshold = 0
    color.printRed('threshold, '+str(threshold) + 
        ' generatition_frequency, '+str(generatition_frequency)
        +' validation_frequency, '+str(validation_frequency))
    done_looping = False
    decay_epochs=500
    n_epochs=600

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
        train_error = 0
        train_lowerbound = 0
        train_hinge_loss = 0
        train_obj = 0
        
        test_epoch = epoch - decay_epochs
        if test_epoch > 0 and test_epoch % 10 == 0:
            print l_r.get_value()
            with open(logdir+'hook.txt', 'a') as f:
                print >>f,l_r.get_value()
            l_r.set_value(np.cast['float32'](l_r.get_value()/3.0))

        tmp_start1 = time.clock()
        for minibatch_index in xrange(n_train_batches):
            #print n_train_batches
            e, l, h, o = train_model(minibatch_index)
            train_error += e
            train_lowerbound += l
            train_hinge_loss += h
            train_obj += o
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

        
        if math.isnan(train_lowerbound):
            NaN_count+=1
            color.printRed("NaN detected. Reverting to saved best parameters")
            print '---------------NaN_count:', NaN_count
            with open(logdir+'hook.txt', 'a') as f:
                print >>f, '---------------NaN_count:', NaN_count
            
            tmp =  [debug_model(i) for i in xrange(n_train_batches)]
            tmp = (np.asarray(tmp)).mean(axis=0) / float(batch_size)
            tmp[0]*=batch_size
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
            tmp = (np.asarray(tmp)).mean(axis=0) / float(batch_size)
            tmp[0]*=batch_size
            print '------------------', tmp
            continue

        n_train=n_train_batches*batch_size
        #print 'optimization_time', time.clock() - tmp_start1
        print epoch, 'stochastic training error', train_error / float(batch_size), train_lowerbound / float(n_train), train_hinge_loss / float(n_train), train_obj / float(n_train)
        with open(logdir+'hook.txt', 'a') as f:
            print >>f, epoch, 'stochastic training error', train_error / float(batch_size), train_lowerbound / float(n_train), train_hinge_loss / float(n_train), train_obj / float(n_train)

        if epoch % validation_frequency == 0:
            tmp_start2 = time.clock()
            # compute zero-one loss on validation set
            #train_stats = [train_bound_without_dropout(i) for i
            #                     in xrange(n_train_batches)]
            #this_train_stats = np.mean(train_stats, axis=0)
            #this_train_stats[1:] = this_train_stats[1:]/ float(batch_size)

            test_stats = [test_model(i) for i in xrange(n_test_batches)]
            this_test_stats = np.mean(test_stats, axis=0)
            this_test_stats[1:] = this_test_stats[1:]/ float(batch_size)
            
            print epoch, 'test error', this_test_stats
            with open(logdir+'hook.txt', 'a') as f:
                print >>f, epoch, 'test error', this_test_stats

        if epoch%100==0:
            model = parameters()
            for i in xrange(len(model)):
                model[i] = np.asarray(model[i]).astype(np.float32)
                #print model[i].shape, np.mean(model[i]), np.var(model[i])
                            
            np.savez(logdir+'model-'+str(epoch), model=model)
                
        
        tmp_start4=time.clock()
        if epoch % generatition_frequency == 0:
            tail='-'+str(epoch)+'.png'
            random_z = np.random.standard_normal((n_batch, n_hidden[-1])).astype(np.float32)
            _x_mean, _x = random_generation(random_z)
            #print _x.shape
            #print _x_mean.shape
            image = paramgraphics.mat_to_img(_x.T, dim_input, colorImg=colorImg)
            image.save(logdir+'samples'+tail, 'PNG')
            image = paramgraphics.mat_to_img(_x_mean.T, dim_input, colorImg=colorImg)
            image.save(logdir+'mean_samples'+tail, 'PNG')
        #print 'generation_time', time.clock() - tmp_start4
        

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
    learning_rate=1e-3
    if os.environ.has_key('learning_rate'):
        learning_rate = float(os.environ['learning_rate'])
    dropout_flag = 1
    if os.environ.has_key('dropout_flag'):
        dropout_flag = int(os.environ['dropout_flag'])
    drop_inverses_flag = 0
    if os.environ.has_key('drop_inverses_flag'):
        drop_inverses_flag = int(os.environ['drop_inverses_flag'])

    cmmva_6layer_dropout_mnist_60000(drop_inverses_flag=drop_inverses_flag,
        dropout_flag=dropout_flag, predir=predir,learning_rate=learning_rate)