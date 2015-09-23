import cPickle
import gzip
import os
import sys
import time
import math
import numpy as np
import scipy.io as sio
import theano
import theano.tensor as T


def labelToMat(y):
    label = np.unique(y)
    newy = np.zeros((len(y), len(label))).astype(np.int32)
    for i in range(len(y)):
        newy[i, y[i]] = 1
    return newy

def load_max_min(dirs='data_imputation/', dataset='svhnlcn', pertub_prob=6):
    #print 'load max min'
    zz = sio.loadmat(dirs+dataset+'_params_'+str(int(pertub_prob))+'_max_min_pixel.mat')
    pixel_max = zz['pixel_max'].flatten()
    pixel_min = zz['pixel_min'].flatten()
    #print pixel_max.shape
    #print pixel_min.shape
    return pixel_max, pixel_min

def load_pertub_data(dirs='data_imputation/', pertub_type=3, pertub_prob=6, pertub_prob1=0, random_guess=False):
    # perturb data
    print 'Loading perturbed data...'

    if pertub_type==4:
        zz = sio.loadmat(dirs+'type_'+str(pertub_type)+'_params_'+str(int(pertub_prob*100))+'_noise_rawdata.mat')
    elif pertub_type==3:
        pertub_prob = int(pertub_prob)
        zz = sio.loadmat(dirs+'type_'+str(pertub_type)+'_params_'+str(pertub_prob)+'_noise_rawdata.mat')
    elif pertub_type==5:
        start = int(pertub_prob)
        end = int(pertub_prob1)
        zz = sio.loadmat(dirs+'type_'+str(pertub_type)+'_params_'+str(start)+'_'+str(end)+'_noise_rawdata.mat')
    else:
        print 'Error in load_pertub_data'
        print dirs, pertub_type, pertub_prob
        exit()

    data_train = zz['z_train'].T
    data = zz['z_test_original'].T
    data_perturbed = zz['z_test'].T
    pertub_label = zz['pertub_label'].astype(np.float32).T
    pertub_number = float(np.sum(1-pertub_label))

    if random_guess:
        # compute the std of known pixels
        mean = data_perturbed.sum()/pertub_number
        std = np.sqrt(np.sum(((data_perturbed - mean)**2)*pertub_label)/pertub_number)
        print 'mean and std for known pixels', mean, std
        # sample from Guassian for random guess
        guess = np.randomnormal(mean, std, data_perturbed.shape)
        data_perturbed = guess*(1-pertub_label) + data_perturbed*pertub_label

    print pertub_number, data_train.shape, data.shape, data_perturbed.shape, pertub_label.shape

    mse = ((data - data_perturbed)**2).sum() / pertub_number
    #print '-------Initial MSE-------', mse

    data_train = theano.shared(np.asarray(data_train, dtype=theano.config.floatX), borrow=True)
    data = theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=True)
    data_perturbed = theano.shared(np.asarray(data_perturbed, dtype=theano.config.floatX), borrow=True)
    pertub_label = theano.shared(np.asarray(pertub_label, dtype=theano.config.floatX), borrow=True)

    return data_train, data, data_perturbed, pertub_label, pertub_number

def load_pertub_data_svhn(dirs='data_imputation/', dataset='svhnlcn', pertub_type=3, pertub_prob=6, pertub_prob1=16, random_guess=True):
    # perturb data
    print 'Loading perturbed data...'

    if pertub_type==4:
        zz = sio.loadmat(dirs+dataset+'_type_'+str(pertub_type)+'_params_'+str(int(pertub_prob*100))+'_noise_rawdata.mat')
    elif pertub_type==3:
        pertub_prob = int(pertub_prob)
        zz = sio.loadmat(dirs+dataset+'_type_'+str(pertub_type)+'_params_'+str(pertub_prob)+'_noise_rawdata.mat')
    elif pertub_type==5:
        pertub_prob = int(pertub_prob)
        pertub_prob1 = int(pertub_prob1)
        zz = sio.loadmat(dirs+dataset+'_type_'+str(pertub_type)+'_params_'+str(pertub_prob)+'_'+str(pertub_prob1)+'_noise_rawdata.mat')
    else:
        print 'Error in load_pertub_data'
        print dirs, pertub_type, pertub_prob
        exit()

    #data_train = zz['z_train'].T
    data = zz['z_test_original'].T
    data_perturbed = zz['z_test'].T
    pertub_label = zz['pertub_label'].astype(np.float32).T
    pertub_number = float(np.sum(1-pertub_label))

    if random_guess:
        # compute the std of known pixels
        mean = data_perturbed.sum()/pertub_number
        std = np.sqrt(np.sum(((data_perturbed - mean)**2)*pertub_label)/pertub_number)
        print 'mean and std for known pixels', mean, std
        # sample from Guassian for random guess
        guess = np.random.normal(mean, std, data_perturbed.shape)
        data_perturbed = guess*(1-pertub_label) + data_perturbed*pertub_label

    print pertub_number, data.shape, data_perturbed.shape, pertub_label.shape

    mse = ((data - data_perturbed)**2).sum() / pertub_number
    #print '-------Initial MSE-------', mse

    #data_train = theano.shared(np.asarray(data_train, dtype=theano.config.floatX), borrow=True)
    data = theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=True)
    data_perturbed = theano.shared(np.asarray(data_perturbed, dtype=theano.config.floatX), borrow=True)
    pertub_label = theano.shared(np.asarray(pertub_label, dtype=theano.config.floatX), borrow=True)

    return data, data_perturbed, pertub_label, pertub_number

def load_pertub_data_cifar(dirs='data_imputation/', dataset='cifar10_gcn_var', pertub_type=3, pertub_prob=6):
    # perturb data
    print 'Loading perturbed data...'

    if pertub_type==4:
        zz = sio.loadmat(dirs+dataset+'_type_'+str(pertub_type)+'_params_'+str(int(pertub_prob*100))+'_noise_rawdata.mat')
    elif pertub_type==3:
        pertub_prob = int(pertub_prob)
        zz = sio.loadmat(dirs+dataset+'_type_'+str(pertub_type)+'_params_'+str(pertub_prob)+'_noise_rawdata.mat')
    elif pertub_type==5:
        zz = sio.loadmat(dirs+dataset+'_type_'+str(pertub_type)+'_params_noise_rawdata.mat')
    else:
        print 'Error in load_pertub_data'
        print dirs, pertub_type, pertub_prob
        exit()

    data_train = zz['z_train'].T
    data = zz['z_test_original'].T
    data_perturbed = zz['z_test'].T
    pertub_label = zz['pertub_label'].astype(np.float32).T
    pertub_number = float(np.sum(1-pertub_label))

    print pertub_number, data_train.shape, data.shape, data_perturbed.shape, pertub_label.shape

    data_train = theano.shared(np.asarray(data_train, dtype=theano.config.floatX), borrow=True)
    data = theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=True)
    data_perturbed = theano.shared(np.asarray(data_perturbed, dtype=theano.config.floatX), borrow=True)
    pertub_label = theano.shared(np.asarray(pertub_label, dtype=theano.config.floatX), borrow=True)

    return data_train, data, data_perturbed, pertub_label, pertub_number


def load_data_gpu(dataset, have_matrix = False):
    ''' 
    Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    #print data_dir, data_file
    #exit()
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        #os.mknod(dataset)
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        """ 
        Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy

        y_matrix = labelToMat(np.asarray(data_y))
        shared_y_matrix = theano.shared(y_matrix)

        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        '''
        print data_x.shape
        print np.max(data_x)
        print np.min(data_x)
        print np.mean(data_x)
        print data_y.shape
        print y_matrix.shape
        print 'Verify y_matrix:', (np.argmax(y_matrix, axis=1) == data_y).sum()
        '''

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_y_matrix, 'int32')

    test_set_x, test_set_y, test_set_matrix = shared_dataset(test_set)
    valid_set_x, valid_set_y, valid_set_matrix = shared_dataset(valid_set)
    train_set_x, train_set_y, train_set_matrix = shared_dataset(train_set)

    if have_matrix:
        rval = [(train_set_x, train_set_y, train_set_matrix), (valid_set_x, valid_set_y, valid_set_matrix), (test_set_x, test_set_y, test_set_matrix)]
    else:
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def load_data_cifar(dataset, have_matrix = False):
    ''' 
    Loads the dataset
    '''
    data = np.load('./data/'+dataset+'.npz')
    train_x, train_y, test_x, test_y = data['data']
    print 'data---', dataset
    print train_x.shape
    print train_x.max()
    print train_x.min()
    print (train_x.mean(axis=1)).max()
    print (train_x.mean(axis=1)).min()
    print test_x.shape
    print test_x.max()
    print test_x.min()
    print (test_x.mean(axis=1)).max()
    print (test_x.mean(axis=1)).min()

    def shared_dataset(data_xy, borrow=True):
        """ 
        Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy

        y_matrix = labelToMat(np.asarray(data_y))
        shared_y_matrix = theano.shared(y_matrix)

        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        '''
        print data_x.shape
        print np.max(data_x)
        print np.min(data_x)
        print np.mean(data_x)
        print data_y.shape
        print y_matrix.shape
        print 'Verify y_matrix:', (np.argmax(y_matrix, axis=1) == data_y).sum()
        '''

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_y_matrix, 'int32')

    test_set_x, test_set_y, test_set_matrix = shared_dataset([test_x, test_y])
    train_set_x, train_set_y, train_set_matrix = shared_dataset([train_x, train_y])

    if have_matrix:
        rval = [(train_set_x, train_set_y, train_set_matrix), (test_set_x, test_set_y, test_set_matrix)]
    else:
        rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval

def load_data_svhn_features(dirs,have_matrix=False):
    ''' 
    Loads the dataset
    '''

    print 'load y'
    f = file("data/SVHN/MYDATA/svhnlcn.bin","rb")
    train_x = np.load(f)
    train_y = np.load(f)
    valid_x = np.load(f)
    valid_y = np.load(f)
    test_x = np.load(f)
    test_y = np.load(f)
    f.close()
    del train_x, valid_x, test_x
    train_y = train_y[:598*1000]
    test_y = test_y[:26000]
    print train_y.shape
    print test_y.shape
    print valid_y.shape
    print 'load test x'
    f = file(dirs+"svhn_features.bin","rb")
    valid_x = np.load(f)
    test_x = np.load(f)
    f.close()
    test_x = test_x[:26000]
    print valid_x.shape
    print test_x.shape
    #print 'create x'
    train_x = np.ones((598*1000,96*4*4))
    print 'load train_x'
    f = file(dirs+"svhn_train_features.bin","rb")
    for i in xrange(598):
        #print i
        train_x[i*1000:(i+1)*1000] = np.load(f) 
    f.close()
    print train_x.shape

    def shared_dataset(data_xy, borrow=True):
        """ 
        Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        y_matrix = labelToMat(np.asarray(data_y))
        shared_y_matrix = theano.shared(y_matrix)
        #print data_y[:5]
        #print y_matrix[:5,:]
        
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        '''
        print data_x.shape
        print np.max(data_x)
        print np.min(data_x)
        print np.mean(data_x)
        print data_y.shape
        print y_matrix.shape
        print 'Verify y_matrix:', (np.argmax(y_matrix, axis=1) == data_y).sum()
        '''

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_y_matrix, 'int32')

    test_set_x, test_set_y, test_set_matrix = shared_dataset([test_x, test_y])
    train_set_x, train_set_y, train_set_matrix = shared_dataset([train_x, train_y])
    valid_set_x, valid_set_y, valid_set_matrix = shared_dataset([valid_x, valid_y])

    if have_matrix:
        rval = [(train_set_x, train_set_y, train_set_matrix), (test_set_x, test_set_y, test_set_matrix), (valid_set_x, valid_set_y, valid_set_matrix)]
    else:
        rval = [(train_set_x, train_set_y), (test_set_x, test_set_y), (valid_set_x, valid_set_y)]    
    return rval

def load_data_svhn(dataset,have_matrix=False):
    ''' 
    Loads the dataset
    '''
    f = file("data/SVHN/MYDATA/"+dataset+".bin","rb")
    train_x = np.load(f)
    train_y = np.load(f)
    valid_x = np.load(f)
    valid_y = np.load(f)
    test_x = np.load(f)
    test_y = np.load(f)
    f.close()
    '''
    print 'data---', dataset
    print train_x.shape
    print train_y.shape
    print train_x.max()
    print train_x.min()
    print (train_x.mean(axis=1)).max()
    print (train_x.mean(axis=1)).min()
    print test_x.shape
    print test_y.shape
    print test_x.max()
    print test_x.min()
    print (test_x.mean(axis=1)).max()
    print (test_x.mean(axis=1)).min()
    print valid_x.shape
    print valid_y.shape
    print valid_x.max()
    print valid_x.min()
    print (valid_x.mean(axis=1)).max()
    print (valid_x.mean(axis=1)).min()
    '''

    def shared_dataset(data_xy, borrow=True):
        """ 
        Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        y_matrix = labelToMat(np.asarray(data_y))
        shared_y_matrix = theano.shared(y_matrix)
        #print data_y[:5]
        #print y_matrix[:5,:]
        
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        '''
        print data_x.shape
        print np.max(data_x)
        print np.min(data_x)
        print np.mean(data_x)
        print data_y.shape
        print y_matrix.shape
        print 'Verify y_matrix:', (np.argmax(y_matrix, axis=1) == data_y).sum()
        '''

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_y_matrix, 'int32')

    test_set_x, test_set_y, test_set_matrix = shared_dataset([test_x, test_y])
    train_set_x, train_set_y, train_set_matrix = shared_dataset([train_x, train_y])
    valid_set_x, valid_set_y, valid_set_matrix = shared_dataset([valid_x, valid_y])

    if have_matrix:
        rval = [(train_set_x, train_set_y, train_set_matrix), (test_set_x, test_set_y, test_set_matrix), (valid_set_x, valid_set_y, valid_set_matrix)]
    else:
        rval = [(train_set_x, train_set_y), (test_set_x, test_set_y), (valid_set_x, valid_set_y)]    
    return rval


def load_data_norb(dataset, To_float=True, have_matrix=False):
    ''' 
    Loads the dataset
    '''
    f = file("data/"+dataset+".bin","rb")
    train_x = np.load(f)
    train_y = np.load(f)
    test_x = np.load(f)
    test_y = np.load(f)
    f.close()
    print 'To_float', To_float
    if To_float:
        train_x = train_x/255.0
        test_x = test_x/255.0
    print 'data---', dataset
    print train_x.shape
    print train_x.max()
    print train_x.min()
    print (train_x.mean(axis=1)).max()
    print (train_x.mean(axis=1)).min()
    print test_x.shape
    print test_x.max()
    print test_x.min()
    print (test_x.mean(axis=1)).max()
    print (test_x.mean(axis=1)).min()
    

    def shared_dataset(data_xy, borrow=True):
        """ 
        Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy

        y_matrix = labelToMat(np.asarray(data_y))
        shared_y_matrix = theano.shared(y_matrix)

        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        '''
        print data_x.shape
        print np.max(data_x)
        print np.min(data_x)
        print np.mean(data_x)
        print data_y.shape
        print y_matrix.shape
        print 'Verify y_matrix:', (np.argmax(y_matrix, axis=1) == data_y).sum()
        '''

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_y_matrix, 'int32')

    test_set_x, test_set_y, test_set_matrix = shared_dataset([test_x, test_y])
    train_set_x, train_set_y, train_set_matrix = shared_dataset([train_x, train_y])

    if have_matrix:
        rval = [(train_set_x, train_set_y, train_set_matrix), (test_set_x, test_set_y, test_set_matrix)]
    else:
        rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval

def load_data_gpu_60000(dataset, have_matrix = False):
    ''' 
    Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    #print data_dir, data_file
    #exit()
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        #os.mknod(dataset)
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        """ 
        Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy

        y_matrix = labelToMat(np.asarray(data_y))
        shared_y_matrix = theano.shared(y_matrix)

        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        '''
        print data_x.shape
        print np.max(data_x)
        print np.min(data_x)
        print np.mean(data_x)
        print data_y.shape
        print y_matrix.shape
        print 'Verify y_matrix:', (np.argmax(y_matrix, axis=1) == data_y).sum()
        '''

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_y_matrix, 'int32')

    test_set_x, test_set_y, test_set_matrix = shared_dataset(test_set)
    valid_set_x, valid_set_y, valid_set_matrix = shared_dataset(valid_set)
    x1,y1=train_set
    x2,y2=valid_set
    #print x1.shape
    #print y1.shape
    #print x2.shape
    #print y2.shape
    x3 = np.vstack((x1,x2))
    y3 = np.hstack((y1,y2))
    #print x3.shape
    #print y3.shape
    #exit()
    train_set_x, train_set_y, train_set_matrix = shared_dataset([x3, y3])

    if have_matrix:
        rval = [(train_set_x, train_set_y, train_set_matrix), (valid_set_x, valid_set_y, valid_set_matrix), (test_set_x, test_set_y, test_set_matrix)]
    else:
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval


def load_feature_gpu_cifar10(dir,start=0,end=96):
    
    train_set = np.load(dir+'train_features.npy')
    #valid_set = np.load(dir+'valid_features.npy')
    test_set = np.load(dir+'test_features.npy')
    train_set = train_set[:,start:end]
    #valid_set = valid_set[:,start:end]
    test_set = test_set[:,start:end]

    def shared_dataset(data_x, borrow=True):
        """ 
        Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        #print data_x.shape

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x

    test_set_x = shared_dataset(test_set)
    #valid_set_x = shared_dataset(valid_set)
    train_set_x = shared_dataset(train_set)

    rval = [train_set_x, test_set_x]
    return rval

def load_feature_gpu(dir,start=500,end=1000):
    
    train_set = np.load(dir+'train_features.npy')
    valid_set = np.load(dir+'valid_features.npy')
    test_set = np.load(dir+'test_features.npy')
    train_set = train_set[:,start:end]
    valid_set = valid_set[:,start:end]
    test_set = test_set[:,start:end]

    def shared_dataset(data_x, borrow=True):
        """ 
        Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        #print data_x.shape

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x

    test_set_x = shared_dataset(test_set)
    valid_set_x = shared_dataset(valid_set)
    train_set_x = shared_dataset(train_set)

    rval = [train_set_x, valid_set_x, test_set_x]
    return rval


def load_mat_gpu(dir):
    result = sio.loadmat(dir)
    train_set =  result['z_train'].T
    valid_set = result['z_test'].T
    test_set = result['z_test'].T

    def shared_dataset(data_x, borrow=True):
        """ 
        Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        #print data_x.shape

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x

    test_set_x = shared_dataset(test_set)
    valid_set_x = shared_dataset(valid_set)
    train_set_x = shared_dataset(train_set)

    rval = [train_set_x, valid_set_x, test_set_x]
    return rval