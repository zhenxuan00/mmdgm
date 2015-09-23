import scipy.io as io
import numpy as np
import cPickle
import os, sys
import scipy.io as sio
import theano
import theano.tensor as T
import pylearn2.expr.preprocessing as pypp
import pylearn2.datasets.norb as norb
import PIL.Image
import pylab
from util import datapy, color, paramgraphics
from random import shuffle
from util import lcn

def perform_lcn(saveDir,strl,x, y):
    n_channels=(0,1,2)
    dim_input = (32, 32)
    colorImg = True
    
    x = x.astype(np.float32)

    print x.shape
    print x.max()
    print x.min()
    print np.max(np.mean(x, axis=1))
    print np.min(np.mean(x, axis=1))
    print strl
    print y[:10]
    print y[40:50]

    image = paramgraphics.mat_to_img(x[:100,:].T, dim_input, colorImg=colorImg, scale=True)
    image.save(saveDir+'svhn_before_lcn_gcn_norm_'+strl+'.png', 'PNG')

    #flatten->'b,c,0,1'->'b,0,1,c'
    x = x.reshape(-1,3,32,32)
    x = np.swapaxes(x, 1, 2)
    x = np.swapaxes(x, 2, 3)
    lcn.transform(x=x,channels=n_channels,img_shape=dim_input)
    #'b,0,1,c'->'b,c,0,1'->flatten
    print x.shape
    x = np.swapaxes(x, 2, 3)
    x = np.swapaxes(x, 1, 2)
    x = x.reshape((-1,32*32*3))
    print x.max()
    print x.min()
    print np.max(np.mean(x, axis=1))
    print np.min(np.mean(x, axis=1))
    image = paramgraphics.mat_to_img(x[:100,:].T, dim_input, colorImg=colorImg, scale=True)
    image.save(saveDir+'svhn_after_lcn_gcn_norm_'+strl+'.png', 'PNG')
    return x

saveDir = 'data/SVHN/MYDATA/'
f = file("data/SVHN/MYDATA/svhngcn_norm.bin","rb")
train_x = np.load(f)
train_y = np.load(f)
valid_x = np.load(f)
valid_y = np.load(f)
test_x = np.load(f)
test_y = np.load(f)
f.close()
valid_x = perform_lcn(saveDir,'valid', valid_x, valid_y)
test_x = perform_lcn(saveDir,'test', test_x, test_y)
train_x = perform_lcn(saveDir,'train', train_x, train_y)

f = file(saveDir+"svhnlcn.bin","wb")
np.save(f,train_x)
np.save(f,train_y)
np.save(f,valid_x)
np.save(f,valid_y)
np.save(f,test_x)
np.save(f,test_y)
f.close()

f = file(saveDir+"svhnlcn_only_test_for_imputation.bin","wb")
np.save(f,test_x)
np.save(f,test_y)
f.close()