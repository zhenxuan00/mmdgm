'''
modified by Chongxuan Li (chongxuanli1991@gmail.com)
'''

import numpy as np
import anglepy
import anglepy.ndict as ndict
from anglepy.models import GPUVAE_YZ_X
import sys, os
import color
import scipy.io as sio


def labelToMat(y):
    label = np.unique(y)
    newy = np.zeros((len(y), len(label)))
    for i in range(len(y)):
        newy[i, y[i]] = 1
    return newy.T


# Load MNIST data
dataset = sys.argv[1]
dir = 'models/mnist_yz_x_50-500-500/'

if len(sys.argv) >= 3:
    dir = sys.argv[2]
print dir

if dataset == 'mnist':
    import anglepy.data.mnist as mnist
    _, train_y, _, _, test_x, test_y = mnist.load_numpy(size=28, binarize_y=False)
    
    if os.environ.has_key('prior') and bool(int(os.environ['prior'])) == True:
        color.printBlue('Have informative prior')
        n_z = 96
        mnist_prior = sio.loadmat('data/mnist_prior/mnist_prior.mat')
        prior_type = os.environ['prior_type']
        color.printBlue('Prior type: '+prior_type)
        if prior_type == 'too_strong':
            test_mean_prior = mnist_prior['z_test']
        elif prior_type == 'naive':
            test_mean_prior = np.zeros((n_z,test_x.shape[1]))
        elif prior_type == 'train_mean':
            train_mean = np.mean(mnist_prior['z_train'], 1)
            test_mean_prior = np.tile(train_mean[:,np.newaxis], [1,test_x.shape[1]])
        elif prior_type == 'test_mean':
            test_mean = np.mean(mnist_prior['z_test'], 1)
            test_mean_prior = np.tile(test_mean[:,np.newaxis], [1,test_x.shape[1]])
        
        print test_mean_prior.shape
        
    else:
        color.printBlue('Have standard prior')
        n_z = 50
        test_mean_prior = np.zeros((n_z,test_x.shape[1]))
        
        
    # Compute prior probabilities per class
    train_y = mnist.binarize_labels(train_y)
    prior_y = train_y.mean(axis=1).reshape((10,1))

    # Create model
    n_x = 28*28
    n_y = 10
    n_hidden = 500,500
    updates = None
    print 'n_z:', n_z
    model = GPUVAE_YZ_X(updates, n_x, n_y, n_hidden, n_z, n_hidden, 'softplus', 'softplus', type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1, uniform_y=True)

    # Load parameters
    ndict.set_value(model.v, ndict.loadz(dir+'v_best.ndict.tar.gz'))
    ndict.set_value(model.w, ndict.loadz(dir+'w_best.ndict.tar.gz'))

elif dataset == 'mnist_basic':
    data_dir = os.environ['ML_DATA_PATH']+'/mnist_variations/'+'mnist_'
    
    tmp = sio.loadmat(data_dir+'train.mat')
    #color.printRed(data_dir+'train.mat')
    train_x = tmp['x_train'].T
    train_y = tmp['t_train'].T.astype(np.int32)
    # validation 2000
    valid_x = train_x[:,10000:]
    valid_y = train_y[10000:]
    train_x = train_x[:,:10000]
    train_y = train_y[:10000]
    tmp = sio.loadmat(data_dir+'test.mat')
    test_x = tmp['x_test'].T
    test_y = tmp['t_test'].T.astype(np.int32)
    test_y = test_y[:,0]
    

    color.printBlue('Have standard prior')
    n_z = 50
    test_mean_prior = np.zeros((n_z,test_x.shape[1]))
        
        
    # Compute prior probabilities per class
    train_y = labelToMat(train_y)
    prior_y = train_y.mean(axis=1).reshape((10,1))

    # Create model
    n_x = 28*28
    n_y = 10
    n_hidden = 500,500
    updates = None
    print 'n_z:', n_z
    model = GPUVAE_YZ_X(updates, n_x, n_y, n_hidden, n_z, n_hidden, 'softplus', 'softplus', type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1, uniform_y=True)

    # Load parameters
    ndict.set_value(model.v, ndict.loadz(dir+'v_best.ndict.tar.gz'))
    ndict.set_value(model.w, ndict.loadz(dir+'w_best.ndict.tar.gz'))
elif dataset == 'mnist_rot':
    data_dir = os.environ['ML_DATA_PATH']+'/mnist_variations/'+'mnist_all_rotation_normalized_float_'
    
    tmp = sio.loadmat(data_dir+'train.mat')
    #color.printRed(data_dir+'train.mat')
    train_x = tmp['x_train'].T
    train_y = tmp['t_train'].T.astype(np.int32)
    # validation 2000
    valid_x = train_x[:,10000:]
    valid_y = train_y[10000:]
    train_x = train_x[:,:10000]
    train_y = train_y[:10000]
    tmp = sio.loadmat(data_dir+'test.mat')
    test_x = tmp['x_test'].T
    test_y = tmp['t_test'].T.astype(np.int32)
    test_y = test_y[:,0]

    color.printBlue('Have standard prior')
    n_z = 50
    test_mean_prior = np.zeros((n_z,test_x.shape[1]))
        
        
    # Compute prior probabilities per class
    train_y = labelToMat(train_y)
    prior_y = train_y.mean(axis=1).reshape((10,1))

    # Create model
    n_x = 28*28
    n_y = 10
    n_hidden = 500,500
    updates = None
    print 'n_z:', n_z
    model = GPUVAE_YZ_X(updates, n_x, n_y, n_hidden, n_z, n_hidden, 'softplus', 'softplus', type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1, uniform_y=True)

    # Load parameters
    ndict.set_value(model.v, ndict.loadz(dir+'v_best.ndict.tar.gz'))
    ndict.set_value(model.w, ndict.loadz(dir+'w_best.ndict.tar.gz'))
elif dataset == 'mnist_back_rand':
    data_dir = os.environ['ML_DATA_PATH']+'/mnist_variations/'+'mnist_background_random_'
    
    tmp = sio.loadmat(data_dir+'train.mat')
    #color.printRed(data_dir+'train.mat')
    train_x = tmp['x_train'].T
    train_y = tmp['t_train'].T.astype(np.int32)
    # validation 2000
    valid_x = train_x[:,10000:]
    valid_y = train_y[10000:]
    train_x = train_x[:,:10000]
    train_y = train_y[:10000]
    tmp = sio.loadmat(data_dir+'test.mat')
    test_x = tmp['x_test'].T
    test_y = tmp['t_test'].T.astype(np.int32)
    test_y = test_y[:,0]

    color.printBlue('Have standard prior')
    n_z = 50
    test_mean_prior = np.zeros((n_z,test_x.shape[1]))
        
        
    # Compute prior probabilities per class
    train_y = labelToMat(train_y)
    prior_y = train_y.mean(axis=1).reshape((10,1))

    # Create model
    n_x = 28*28
    n_y = 10
    n_hidden = 500,500
    updates = None
    print 'n_z:', n_z
    model = GPUVAE_YZ_X(updates, n_x, n_y, n_hidden, n_z, n_hidden, 'softplus', 'softplus', type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1, uniform_y=True)

    # Load parameters
    ndict.set_value(model.v, ndict.loadz(dir+'v_best.ndict.tar.gz'))
    ndict.set_value(model.w, ndict.loadz(dir+'w_best.ndict.tar.gz'))
elif dataset == 'rectangle_image':
    data_dir = os.environ['ML_DATA_PATH']+'/mnist_variations/'+'rectangles_im_'
    
    tmp = sio.loadmat(data_dir+'train.mat')
    #color.printRed(data_dir+'train.mat')
    train_x = tmp['x_train'].T
    train_y = tmp['t_train'].T.astype(np.int32)
    # validation 2000
    valid_x = train_x[:,10000:]
    valid_y = train_y[10000:]
    train_x = train_x[:,:10000]
    train_y = train_y[:10000]
    tmp = sio.loadmat(data_dir+'test.mat')
    test_x = tmp['x_test'].T
    test_y = tmp['t_test'].T.astype(np.int32)
    test_y = test_y[:,0]

    color.printBlue('Have standard prior')
    n_z = 50
    test_mean_prior = np.zeros((n_z,test_x.shape[1]))
        
        
    # Compute prior probabilities per class
    train_y = labelToMat(train_y)
    prior_y = train_y.mean(axis=1).reshape((2,1))

    # Create model
    n_x = 28*28
    n_y = 2
    n_hidden = 500,500
    updates = None
    print 'n_z:', n_z
    model = GPUVAE_YZ_X(updates, n_x, n_y, n_hidden, n_z, n_hidden, 'softplus', 'softplus', type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1, uniform_y=True)

    # Load parameters
    ndict.set_value(model.v, ndict.loadz(dir+'v_best.ndict.tar.gz'))
    ndict.set_value(model.w, ndict.loadz(dir+'w_best.ndict.tar.gz'))

elif dataset == 'mnist_back_image_rot':
    data_dir = os.environ['ML_DATA_PATH']+'/mnist_variations/'+'mnist_all_background_images_rotation_normalized_'
    
    tmp = sio.loadmat(data_dir+'train.mat')
    #color.printRed(data_dir+'train.mat')
    train_x = tmp['x_train'].T
    train_y = tmp['t_train'].T.astype(np.int32)
    # validation 2000
    valid_x = train_x[:,10000:]
    valid_y = train_y[10000:]
    train_x = train_x[:,:10000]
    train_y = train_y[:10000]
    tmp = sio.loadmat(data_dir+'test.mat')
    test_x = tmp['x_test'].T
    test_y = tmp['t_test'].T.astype(np.int32)
    test_y = test_y[:,0]

    color.printBlue('Have standard prior')
    n_z = 50
    test_mean_prior = np.zeros((n_z,test_x.shape[1]))
        
        
    # Compute prior probabilities per class
    train_y = labelToMat(train_y)
    prior_y = train_y.mean(axis=1).reshape((10,1))

    # Create model
    n_x = 28*28
    n_y = 10
    n_hidden = 500,500
    updates = None
    print 'n_z:', n_z
    model = GPUVAE_YZ_X(updates, n_x, n_y, n_hidden, n_z, n_hidden, 'softplus', 'softplus', type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1, uniform_y=True)

    # Load parameters
    ndict.set_value(model.v, ndict.loadz(dir+'v_best.ndict.tar.gz'))
    ndict.set_value(model.w, ndict.loadz(dir+'w_best.ndict.tar.gz'))
    
elif dataset == 'mnist_back_image':
    data_dir = os.environ['ML_DATA_PATH']+'/mnist_variations/'+'mnist_background_images_'
    
    tmp = sio.loadmat(data_dir+'train.mat')
    #color.printRed(data_dir+'train.mat')
    train_x = tmp['x_train'].T
    train_y = tmp['t_train'].T.astype(np.int32)
    # validation 2000
    valid_x = train_x[:,10000:]
    valid_y = train_y[10000:]
    train_x = train_x[:,:10000]
    train_y = train_y[:10000]
    tmp = sio.loadmat(data_dir+'test.mat')
    test_x = tmp['x_test'].T
    test_y = tmp['t_test'].T.astype(np.int32)
    test_y = test_y[:,0]

    color.printBlue('Have standard prior')
    n_z = 50
    test_mean_prior = np.zeros((n_z,test_x.shape[1]))
        
        
    # Compute prior probabilities per class
    train_y = labelToMat(train_y)
    prior_y = train_y.mean(axis=1).reshape((10,1))

    # Create model
    n_x = 28*28
    n_y = 10
    n_hidden = 500,500
    updates = None
    print 'n_z:', n_z
    model = GPUVAE_YZ_X(updates, n_x, n_y, n_hidden, n_z, n_hidden, 'softplus', 'softplus', type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1, uniform_y=True)

    # Load parameters
    ndict.set_value(model.v, ndict.loadz(dir+'v_best.ndict.tar.gz'))
    ndict.set_value(model.w, ndict.loadz(dir+'w_best.ndict.tar.gz'))

else:
    raise Exception("Unknown dataset")

# Make predictions on test set
def get_lowerbound():
    lb = np.zeros((n_y,test_x.shape[1]))
    for _class in range(n_y):
        y = np.zeros((n_y,test_x.shape[1]))
        y[_class,:] = 1
        #_lb = model.eval({'x': test_x.astype(np.float32), 'mean_prior':test_mean_prior.astype(np.float32), 'y':y.astype(np.float32)}, {})
        _lb = model.eval_for_classcondition_prior({'x': test_x.astype(np.float32), 'mean_prior':test_mean_prior.astype(np.float32), 'y':y.astype(np.float32)}, {})
        
        lb[_class,:] = _lb
    return lb

def get_predictions(n_samples=1000, show_convergence=True):
    px = 0
    def get_posterior(likelihood, prior):
        posterior = (likelihood * prior)
        posterior /= posterior.sum(axis=0, keepdims=True)
        return posterior
    for i in range(n_samples):
        px += np.exp(get_lowerbound())
        if show_convergence:
            posterior = get_posterior(px / (i+1), prior_y)
            pred = np.argmax(posterior, axis=0)
            error_perc = 100* (pred != test_y).sum() / (1.*test_y.shape[0])
            print 'samples:', i, ', test-set error (%):', error_perc
    posterior = get_posterior(px / n_samples, prior_y)
    return np.argmax(posterior, axis=0)

n_samples = 1000
print 'Computing class posteriors using a marginal likelihood estimate with importance sampling using ', n_samples, ' samples.'
print 'This is slow, but could be sped up significantly by fitting a classifier to match the posteriors (of the generative model) in the training set.'
result = get_predictions(n_samples)
print 'Done.'
print 'Result (test-set error %): ', result

'''
# Compare predictions with truth
print 'Predicting with 1, 10, 100 and 1000 samples'
for n_samples in [1,10,100,1000]:
    print 'Computing predictions with n_samples = ', n_samples
    predictions = get_predictions(n_samples)
    error_perc = 100* (predictions != test_y).sum() / (1.*test_y.shape[0])
    print 'Error rate is ', error_perc, '%'
'''
