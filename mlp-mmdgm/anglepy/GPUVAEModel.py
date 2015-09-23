'''
modified by Chongxuan Li (chongxuanli1991@gmail.com)
'''

import numpy as np
import pdb
import theano
import theano.tensor as T
import math
import theano.compile
import anglepy.ndict as ndict
from anglepy.misc import lazytheanofunc
import anglepy.logpdfs
import inspect


# ====
# VARIATIONAL AUTO-ENCODER MODEL
# ====

def shared32(x, name=None, borrow=False):
    return theano.shared(np.asarray(x, dtype='float32'), name=name, borrow=borrow)

# Model
class GPUVAEModel(object):
    
    def __init__(self, get_optimizer, theano_warning='raise'):
        
        v = self.v
        w = self.w
        theanofunction = lazytheanofunc('warn', mode='FAST_RUN')
        theanofunction_silent = lazytheanofunc('ignore', mode='FAST_RUN')
        
        # Create theano expressions
        x, z = ndict.ordereddicts(self.variables())
        self.var_x, self.var_z, = x, z
        
        # Helper variables
        A = T.fmatrix('A')
        self.var_A = A
        
        '''
        # Get gradient symbols
        print 'model, x'
        for (d, xx) in x.items():
          print d
          print xx.shape
          
        print x.values()
        '''
        
        allvars = x.values() + z.values() + [A] # note: '+' concatenates lists
        
        # TODO: more beautiful/standardized way of setting distributions
        # (should be even simpler than this) 
        self.dist_qz = {}
        self.dist_px = {}
        self.dist_pz = {}
        
        factors = self.factors(x, z, A)
        if len(factors) == 3:
            (logpx, logpz, logqz) = factors
            cost = 0
            sparsity_penalty = 0
        else:
            (logpx, logpz, logqz, cost, sparsity_penalty) = factors

        if get_optimizer == None:
            def get_optimizer(w, g):
                from collections import OrderedDict
                updates = OrderedDict()
                for i in w: updates[w[i]] = w[i]
                return updates

        # Log-likelihood lower bound
        self.f_L = theanofunction(allvars, [logpx, logpz, logqz, cost, sparsity_penalty])
        L = (logpx + logpz - logqz).sum() - cost - sparsity_penalty

        g = T.grad(L, v.values() + w.values())
        gv, gw = dict(zip(v.keys(), g[0:len(v)])), dict(zip(w.keys(), g[len(v):len(v)+len(w)]))
        updates = get_optimizer(v, gv)
        updates.update(get_optimizer(w, gw))
        
        #self.profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
        #self.f_evalAndUpdate = theano.function(allvars, [logpx + logpz - logqz], updates=updates_w, mode=self.profmode)
        #theano.printing.debugprint(self.f_evalAndUpdate)
        
        self.f_eval_test = theanofunction(allvars, [logpx + logpz - logqz, logpx, logpz, -logqz])
        self.f_eval = theanofunction(allvars, [logpx + logpz - logqz])
        self.f_evalAndUpdate = theanofunction(allvars, [logpx + logpz - logqz], updates=updates)
        self.f_eval_for_classcondition_prior = theanofunction(allvars, [logpx - logqz])
        
    # NOTE: IT IS ESSENTIAL THAT DICTIONARIES OF SYMBOLIC VARS AND RESPECTIVE NUMPY VALUES HAVE THE SAME KEYS
    # (OTHERWISE FUNCTION ARGUMENTS ARE IN INCORRECT ORDER)
    
    def variables(self): raise NotImplementedError()
    def factors(self): raise NotImplementedError()
    def gen_xz(self): raise NotImplementedError()
    def init_w(self): raise NotImplementedError()
    
    # Prediction
    # TODO: refactor to new solution
    def distribution(self, v, w, x, z, name):
        x, z = self.xz_to_theano(x, z)
        v, w, x, z = ndict.ordereddicts((v, w, x, z))
        A = self.get_A(x)
        allvars = v.values() + w.values() + x.values() + z.values() + [A]
        return self.f_dists[name](*allvars)
    
    # Numpy <-> Theano var conversion
    def xz_to_theano(self, x, z): return x, z
    def gw_to_numpy(self, gv, gw): return gv, gw
    
    # A = np.ones((1, n_batch))
    def get_A(self, x): return np.ones((1, x.itervalues().next().shape[1])).astype('float32')

    # Evaluate lower bound
    def eval(self, x, z):
        x, z = self.xz_to_theano(x, z)
        z, x = ndict.ordereddicts((z, x))
        A = self.get_A(x)
        allvars = x.values() + z.values() + [A]
        L = self.f_eval(*allvars)
        return L[0]
        
    # evaluation for classcondtion_prior
    def eval_for_classcondition_prior(self, x, z):
        x, z = self.xz_to_theano(x, z)
        z, x = ndict.ordereddicts((z, x))
        A = self.get_A(x)
        allvars = x.values() + z.values() + [A]
        L = self.f_eval_for_classcondition_prior(*allvars)
        return L[0]
        
    def eval_test(self, x, z):
        x, z = self.xz_to_theano(x, z)
        z, x = ndict.ordereddicts((z, x))
        A = self.get_A(x)
        allvars = x.values() + z.values() + [A]
        L = self.f_eval_test(*allvars)
        return L
        
    # Gradient of lower bound w.r.t. parameters
    def evalAndUpdate(self, x, z):
        x, z = self.xz_to_theano(x, z)
        z, x = ndict.ordereddicts((z, x))
        A = self.get_A(x)
        allvars = x.values() + z.values() + [A]
        L = self.f_evalAndUpdate(*allvars)
        return L[0]

    # Compute likelihood lower bound given a variational auto-encoder
    # L is number of samples
    def est_loglik(self, x, n_batch, n_samples=1, byteToFloat=False):
        
        n_tot = x.itervalues().next().shape[1]
        
        px = 0 # estimate of marginal likelihood
        lowbound = 0 # estimate of lower bound of marginal likelihood
        for _ in range(n_samples):
            _L = np.zeros((1,0))
            i = 0
            while i < n_tot:
                i_to = min(n_tot, i+n_batch)
                _x = ndict.getCols(x, i, i_to)
                if byteToFloat: _x = {i:_x[i].astype(np.float32)/256. for i in _x}
                _L = np.hstack((_L, self.eval(_x, {})))
                i += n_batch
            lowbound += _L.mean()
            px += np.exp(_L)
        
        lowbound /= n_samples
        logpx = np.log(px / n_samples).mean()
        return lowbound, logpx
        
    def test(self, x, n_batch, n_samples=1, byteToFloat=False):
        
        n_tot = x.itervalues().next().shape[1]
        
        px = 0 # estimate of marginal likelihood
        lowbound = 0 # estimate of lower bound of marginal likelihood
        pzzz = 0
        pxxx = 0
        qzzz = 0
        for _ in range(n_samples):
            _L = np.zeros((1,0))
            px = np.zeros((1,0))
            pz = np.zeros((1,0))
            qz = np.zeros((1,0))
            
            i = 0
            while i < n_tot:
                i_to = min(n_tot, i+n_batch)
                _x = ndict.getCols(x, i, i_to)
                if byteToFloat: _x = {i:_x[i].astype(np.float32)/256. for i in _x}
                result = self.eval_test(_x, {})
                _L = np.hstack((_L, result[0]))
                px = np.hstack((px, result[1]))
                pz = np.hstack((pz, result[2]))
                qz = np.hstack((qz, result[3]))
                
                i += n_batch
            lowbound += _L.mean()
            pzzz+= pz.mean()
            pxxx+= px.mean()
            qzzz+= qz.mean()
        
        lowbound /= n_samples
        pzzz /= n_samples
        pxxx /= n_samples
        qzzz /= n_samples
        
        return lowbound, pxxx, pzzz, qzzz