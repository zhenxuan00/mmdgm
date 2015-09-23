'''
Code for mmDGM
Author: Chongxuan Li (chongxuanli1991@gmail.com)
Version = '1.0'
'''

import numpy as np
import os, sys
import time
import pickle, gzip
import scipy.io as sio

class Pegasos:
    """
    a simple implementation of pegasos for multiple classification
    
    pegasos_k: int
        size of the mini batch
        
    pegasos_T: int
        iteration times
        
    pegasos_lambda: float
        trade-off between weight decay and hinge-loss
    
    number_classes: int
        number of possible labels
    
    eta : float matrix [nc dimX]
        weight vectors learned by pegasos
    
    L : float, optional 
        using 0-L loss instead of 0-1 loss in SVM
        
    Check_gradient : binary, flag
        using numeric method to check gradient
        
    total_gradient : float same_size with pegasos
        for adagrad
    
    learning_rate : float
        global learning rate
    """
    
    def __init__(self, pegasos_k = 100, pegasos_T = 500, pegasos_lambda = 0.1, nc = 2, L = 1):
        self.pegasos_T = pegasos_T
        self.pegasos_k = pegasos_k
        self.pegasos_lambda = pegasos_lambda
        self.nc = nc
        self.L = L

    def init_H(self, A_t, y):
        grad = self.computeGradient(A_t, y)
        self.total_gradient += grad**2
        
    def init_param(self, dimX):
        self.eta = np.random.normal(0, 0.01,(self.nc, dimX))
        self.check_gradient = False
        self.check_objective = False
        self.learning_rate = 0.1
        self.total_gradient  = np.zeros((self.nc, dimX))
        
    def pegasos_optimize(self, X, Y, X_t, Y_t):
        # initialize weight as a zero vector
        [N, dimX] = X.shape
        self.init_param(dimX)
        # sample batches
        batches = np.arange(0, N, self.pegasos_k)
        if batches[-1] != N:
            batches = np.append(batches, N)
        #print batches
        
        if self.check_objective:
            self.fp = open("result",'w')
        
        for i in xrange(10):
            ii = i % (len(batches) - 2)
            minibatch = X[batches[ii]:batches[ii + 1]]
            label = Y[batches[ii]:batches[ii + 1]]
            self.init_H(minibatch.T, label)
        
        for j in xrange(self.pegasos_T):
            jj = j % (len(batches) - 2)
            minibatch = X[batches[jj]:batches[jj + 1]]
            label = Y[batches[jj]:batches[jj + 1]]
            
            self.pegasos_iter(minibatch.T, label, j + 1)
            
            # check the value of the objective funtion
            if self.check_objective:
                self.objective(X.T, Y, self.eta)
            
            if ((j+1) % 1000 == 0):
                print "Iteration: ", j + 1, " Testing Score: ", self.pegasos_score(X_t,Y_t)
                #" Training Score: ", self.pegasos_score(X,Y), 
                #print self.eta
            
        if self.check_objective:
            self.fp.close()
        
    def pegasos_iter(self, A_t, y, t):
        # check the gradient using numeric method
        if self.check_gradient:
            print 'Compute numgrad...'
            numgrad = self.numericGradient(self.objective, A_t, y, self.eta);
            
        grad = self.computeGradient(A_t, y)
        
        if False:
            eta_t = 1.0/(t * self.pegasos_lambda)
            self.eta -= eta_t * grad
        else:
            self.total_gradient += grad**2
            self.eta -= self.learning_rate * (grad / (1e-4 + np.sqrt(self.total_gradient)))
        
        if self.check_gradient:
            print 'grad: ', np.sum((grad)**2)
            print 'This relative ratio should be small: ', np.sum((grad-numgrad)**2)/np.sum((grad+numgrad)**2)
    
    def computeGradient(self, A_t, y):
        """
            computeGradient
        """
        
        [dimX, k] = A_t.shape
        
        # generate the 0-1 loss matrix
        l_y = np.ones((self.nc, k))
        l_y[y, xrange(k)] = 0
        l_y = l_y * self.L
        
        # compute the result
        """
            This line is the bottleneck of the algorithm 
        """
        multi_result = self.eta.dot(A_t)
        
        # generate the matrix whose columns are filled with only true label result 
        label_result = multi_result[y, xrange(k)] * np.ones((self.nc,k))
        
        # find the max label
        y_m = np.argmax(l_y - label_result + multi_result, axis = 0)
        
        # compute gradient
        grad = np.zeros(self.eta.shape)
        
        # a vectorization version of updating
        for cc in range(self.nc):
            grad[cc, :] += (A_t[:, y_m == cc]).sum(axis = 1)
            grad[cc, :] -= (A_t[:, y == cc]).sum(axis = 1)
        grad /= k;
        grad += self.eta * self.pegasos_lambda
        
        return grad
        
        
    def objective(self, data, y, eta_x):
        f = self.pegasos_lambda / 2 * np.sum(eta_x**2)
        
        [dimX, m] = data.shape
        
        # generate the 0-L loss matrix
        l_y = np.ones((self.nc, m))
        l_y[y, xrange(m)] = 0
        l_y = l_y * self.L

        # compute the result
        multi_result = eta_x.dot(data)

        # generate the matrix whose columns are filled with only true label result 
        label_result = multi_result[y, xrange(m)] * np.ones((self.nc,m))
        
        # compute the hinge loss
        hinge_loss = np.max(l_y - label_result + multi_result, axis = 0)
        self.fp.writelines(str(f)+' '+str(np.sum(hinge_loss) / m)+'\n') 

        f += np.sum(hinge_loss) / m
        return f
        
    def numericGradient(self, function, data, y, x):
        """
            The function that computes the numeric gradient
            'function' is real-valued function over 'x'
            'x' must be a matrix    
            'numgrad' has the same dimension with 'x'
        """
        
        EPS = 1e-4
        numgrad = np.zeros(x.shape)
        d1 = x.shape[0]
        d2 = x.shape[1]    
        
        for dd1 in xrange(d1):
            for dd2 in xrange(d2):
                tmp = np.zeros(x.shape)
                tmp[dd1,dd2] = EPS
                numgrad[dd1,dd2] = 0.5*(function(data, y, x+tmp)-function(data, y, x-tmp)) / EPS
        return numgrad
    
    def testNum(self,a,b,x):
        value = x[0,0]**2 + 3*x[0,0]*x[0,1]    
        return value
        
    def gradtestNum(self,a,b,x):
        grad = np.zeros(x.shape);
        grad[0,0] = 2*x[0,0] + 3*x[0,1]
        grad[0,1] = 3*x[0,0]
        return grad
        
    def get_eta(self):
        return self.eta
        
    def pegasos_score(self, X, Y):
        predict = np.argmax((self.eta.dot(X.T)), axis = 0)
        result = np.zeros(Y.shape[0])
        result[predict == Y] = 1
        return np.sum(result)/ Y.shape[0]
        
    def pegasos_score_compare(self, X, Y, eta):
        predict = np.argmax((self.eta.dot(X.T)), axis = 0)
        result = np.zeros(Y.shape[0])
        result[predict == Y] = 1
        print np.sum(result)/ Y.shape[0]
        
        predict = np.argmax((eta.dot(X.T)), axis = 0)
        result = np.zeros(Y.shape[0])
        result[predict == Y] = 1
        print np.sum(result)/ Y.shape[0]

if __name__ == "__main__":
    
    dataset = sys.argv[2]
    if (not dataset == 'mnist_binarized') and ('svhn' not in dataset):
        result = sio.loadmat(sys.argv[1])
        # train_data = mat[0][0]
        train_data =  result['z_train'].T
        # test_data = mat[2][0]
        test_data = result['z_test'].T
    if dataset == 'mnist':
        mat = pickle.load(gzip.open('data/mnist/mnist_28.pkl.gz', 'rb'))
        test_label = mat[2][1]
        #print test_label[1:10]
        train_label = mat[0][1]
    elif dataset == 'cifar10':
        result1 = sio.loadmat('data/cifar10_prior/cifar10_prior.mat')
        test_label = result1['test_y'][0,:]
        #print test_label.shape
        train_label = result1['train_y'][0,:]
    elif dataset == 'svhn':
        train_x, train_y, test_x, test_y = np.load('data/svhn/svhn.npy')
        train_data = train_x.T
        test_data = test_x.T
        train_label = train_y.astype(np.int32)
        test_label = test_y.astype(np.int32)
        
    elif dataset == 'svhn_prior':
        train_x, train_y, test_x, test_y = np.load('data/svhn/svhn_prior.npy')
        train_data = train_x.T
        test_data = test_x.T
        train_label = train_y.astype(np.int32)
        test_label = test_y.astype(np.int32)
        
    elif dataset == 'mnist_binarized':
        mat = pickle.load(gzip.open('data/mnist/mnist_28.pkl.gz', 'rb'))
        test_label1 = mat[2][1]
        train_label1 = mat[0][1]
        
        data_dir = '/home/lichongxuan/regbayes2/data/mat_data/binarized_mnist_'
        tmp = sio.loadmat(data_dir+'train_used.mat')
        train_data = tmp['train_x'].astype(np.float32)
        train_label = tmp['train_y'][0,:train_data.shape[0]].astype(np.int32)
        tmp = sio.loadmat(data_dir+'test_used.mat')
        test_data = tmp['test_x'].astype(np.float32)
        test_label = tmp['test_y'][0,:test_data.shape[0]].astype(np.int32)
        print np.sum((test_label1 == test_label).astype(np.int32))
        print np.sum((train_label1 == train_label).astype(np.int32))
    else:
        data_dir = os.environ['ML_DATA_PATH']+'/mnist_variations/'
        if dataset == 'mnist_rot':
            data_dir+='mnist_all_rotation_normalized_float_'
        elif dataset == 'mnist_back_rand':
            data_dir+='mnist_background_random_'
        elif dataset == 'mnist_back_image':
            data_dir+='mnist_background_images_'
        elif dataset == 'mnist_back_image_rot':
            data_dir+='mnist_all_background_images_rotation_normalized_'
        elif dataset == 'rectangle':
            data_dir+='rectangles_'
        elif dataset == 'rectangle_image':
            data_dir+='rectangles_im_'    
        elif dataset == 'convex':
            data_dir+='convex_'    
        elif dataset == 'mnist_basic':
            data_dir+='mnist_'
        else:
            print 'error'
            exit()
            
        tmp = sio.loadmat(data_dir+'train.mat')
        train_label = tmp['t_train'][0,:train_data.shape[0]].astype(np.int32)
        tmp = sio.loadmat(data_dir+'test.mat')
        test_label = tmp['t_test'][0,:test_data.shape[0]].astype(np.int32)
    
        
    
    '''
    print train_data.shape
    print test_data.shape
    print train_label.shape
    print test_label.shape
    '''
    print 'The test score means accuracy'
    
    if dataset == 'mnist':
        nc = 10
        pegasos_batch = 100
        lam = 1E-4
        T = 1 / lam * 20
        T = int(T)
        param = dict()
        param['L'] = 1.0
        param['T'] = T
    else:
        nc = 10
        pegasos_batch = 100
        lam = 1E-4
        T = 1 / lam * 20
        T = int(T)
        param = dict()
        param['L'] = 1.0
        param['T'] = T
        
    if os.environ.has_key('L'):
      param['L'] = int(os.environ['L'])
    if os.environ.has_key('T'):
      param['T'] = int(os.environ['T'])
    param['input'] = sys.argv[1]
    if os.environ.has_key('mark'):
      param['mark'] = os.environ['mark']

    p = Pegasos(pegasos_batch, param['T'] , lam, nc, param['L'])
    p.pegasos_optimize(train_data, train_label, test_data, test_label)
    print str(param), "Testing score: ", p.pegasos_score(test_data, test_label)
    with open('log.txt', 'a') as f:
      print >>f, str(param), "Testing score: ", p.pegasos_score(test_data, test_label)
