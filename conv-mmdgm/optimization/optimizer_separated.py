'''
Different optimizer for minimization
'''

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

def shared32(x, name=None, borrow=False):
    return theano.shared(np.asarray(x, dtype='float32'), name=name, borrow=borrow)


def get_momentum_optimizer_max(learning_rate=0.01, momentum=0.9, weight_decay=0.0):
    print 'momentum', learning_rate.get_value(), momentum, weight_decay
    def get_optimizer(w, g, l, d):
        # Store the parameters in dict or in list
        #updates = OrderedDict()
        updates = []
        
        for i in xrange(len(w)):
            gi = g[i]
            if weight_decay > 0:
              gi -= weight_decay * d[i] * w[i]
            mom = shared32(w[i].get_value() * 0.)
            # Update moments
            mom_new = momentum * mom + learning_rate * l[i] * (1 - momentum) * gi
            # Do update
            w_new = w[i] + mom_new
            updates = updates + [(w[i], w_new),(mom, mom_new)]
        return updates

    return get_optimizer

def get_momentum_optimizer_min(learning_rate=0.01, momentum=0.9, weight_decay=0.0):
    print 'momentum', learning_rate.get_value(), momentum, weight_decay
    def get_optimizer(w, g, l, d):
        # Store the parameters in dict or in list
        #updates = OrderedDict()
        updates = []
        
        for i in xrange(len(w)):
            gi = g[i]
            if weight_decay > 0:
              gi += weight_decay * d[i] * w[i]
            mom = shared32(w[i].get_value() * 0.)
            # Update moments
            mom_new = momentum * mom + learning_rate * l[i] * (1 - momentum) * gi
            # Do update
            w_new = w[i] - mom_new
            updates = updates + [(w[i], w_new),(mom, mom_new)]
        return updates

    return get_optimizer

def get_adam_optimizer_max(learning_rate=0.001, decay1=0.1, decay2=0.001, weight_decay=0.0, epsilon=1e-8):
    '''
    Implementation of AdaM
        All of the parameters are default in the ICLR paper
        Not the exact procedure, no lambda in paper ,even by changing  decay = 1 - beta
        Used for minimization
    '''
    print 'AdaM', learning_rate.get_value(), decay1, decay2, weight_decay, epsilon
    def shared32(x, name=None, borrow=False):
        return theano.shared(np.asarray(x, dtype='float32'), name=name, borrow=borrow)

    def get_optimizer(w, g, l, d):
        # Store the parameters in dict or in dist
        #updates = OrderedDict()
        updates = []
        
        it = shared32(0.)
        ###
        
        #updates[it] = it + 1.
        it_new = it + 1.
        updates+=[(it, it_new)]

        fix1 = 1.-(1.-decay1)**(it+1.) # To make estimates unbiased
        fix2 = 1.-(1.-decay2)**(it+1.) # To make estimates unbiased
        lr_t = learning_rate * T.sqrt(fix2) / fix1
        
        ###
        #print xrange(len(w))
        #for i in w:
        for i in xrange(len(w)):
            
            gi = g[i]
            if weight_decay > 0:
              gi -= weight_decay * d[i] * w[i] #T.tanh(w[i])

            # mean_squared_grad := E[g^2]_{t-1}
            mom1 = shared32(w[i].get_value() * 0.)
            mom2 = shared32(w[i].get_value() * 0.)
            
            # Update moments
            mom1_new = mom1 + decay1 * (gi - mom1)
            mom2_new = mom2 + decay2 * (T.sqr(gi) - mom2)
            
            # Compute the effective gradient and effective learning rate
            effgrad = mom1_new / (T.sqrt(mom2_new) + epsilon)
            
            effstep_new = lr_t * l[i] * effgrad
            
            # Do update
            w_new = w[i] + effstep_new
              
            # Apply update
            
            #updates[w[i]] = w_new
            #updates[mom1] = mom1_new
            #updates[mom2] = mom2_new

            ###

            updates = updates + [(w[i], w_new),(mom1, mom1_new),(mom2, mom2_new)]

        return updates

    return get_optimizer

def get_adam_optimizer_min(learning_rate=0.001, decay1=0.1, decay2=0.001, weight_decay=0.0, epsilon=1e-8):
    '''
    Implementation of AdaM
        All of the parameters are default in the ICLR paper
        Not the exact procedure, no lambda in paper ,even by changing  decay = 1 - beta
        Used for minimization
    '''
    print 'AdaM', learning_rate.get_value(), decay1, decay2, weight_decay, epsilon
    def shared32(x, name=None, borrow=False):
        return theano.shared(np.asarray(x, dtype='float32'), name=name, borrow=borrow)

    def get_optimizer(w, g, l, d):
        # Store the parameters in dict or in dist
        #updates = OrderedDict()
        updates = []
        
        it = shared32(0.)
        ###
        
        #updates[it] = it + 1.
        it_new = it + 1.
        updates+=[(it, it_new)]

        fix1 = 1.-(1.-decay1)**(it+1.) # To make estimates unbiased
        fix2 = 1.-(1.-decay2)**(it+1.) # To make estimates unbiased
        lr_t = learning_rate * T.sqrt(fix2) / fix1
        
        ###
        #print xrange(len(w))
        #for i in w:
        for i in xrange(len(w)):
            
            gi = g[i]
            if weight_decay > 0:
              gi += weight_decay * d[i] * w[i] #T.tanh(w[i])

            # mean_squared_grad := E[g^2]_{t-1}
            mom1 = shared32(w[i].get_value() * 0.)
            mom2 = shared32(w[i].get_value() * 0.)
            
            # Update moments
            mom1_new = mom1 + decay1 * (gi - mom1)
            mom2_new = mom2 + decay2 * (T.sqr(gi) - mom2)
            
            # Compute the effective gradient and effective learning rate
            effgrad = mom1_new / (T.sqrt(mom2_new) + epsilon)
            
            effstep_new = lr_t * l[i] * effgrad
            
            # Do update
            w_new = w[i] - effstep_new
              
            # Apply update
            
            #updates[w[i]] = w_new
            #updates[mom1] = mom1_new
            #updates[mom2] = mom2_new

            ###

            updates = updates + [(w[i], w_new),(mom1, mom1_new),(mom2, mom2_new)]

        return updates

    return get_optimizer

