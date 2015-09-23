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


saveDir = './data/SVHN/MYDATA/'
if not os.path.exists(saveDir): os.makedirs(saveDir)
batch1_data = []
batch1_labels = []
batch2_data = []
batch2_labels = []
dim_input = (32, 32)
n_channels=3
colorImg = True
preprocessing = sys.argv[1]


train = io.loadmat('./data/SVHN/format2/train_32x32.mat')
x = train['X'].transpose((2, 0, 1, 3)).reshape((3072, -1))
y = train['y'].reshape((-1,))
for i in np.arange(len(y)):
    if y[i] == 10:
        y[i] = 0
index = np.arange(len(y))
shuffle(index)
x = x[:, index]
y = y[index]

count = np.zeros((10,), 'int32')
for i in np.arange(len(y)):
    if count[y[i]] < 400:
        count[y[i]] += 1
        batch2_data.append(x[:, i])
        batch2_labels.append(y[i])
    else:
        batch1_data.append(x[:, i])
        batch1_labels.append(y[i])

print '---train'
extra = io.loadmat('./data/SVHN/format2/extra_32x32.mat')
x = extra['X'].transpose((2, 0, 1, 3)).reshape((3072, -1))
y = extra['y'].reshape((-1,))
del extra
for i in np.arange(len(y)):
    if y[i] == 10:
        y[i] = 0
index = np.arange(len(y))
shuffle(index)
x = x[:, index]
y = y[index]

count = np.zeros((10,), 'int32')
for i in np.arange(len(y)):
    if count[y[i]] < 200:
        count[y[i]] += 1
        batch2_data.append(x[:, i])
        batch2_labels.append(y[i])
    else:
        batch1_data.append(x[:, i])
        batch1_labels.append(y[i])
batch1_data = np.asarray(batch1_data)
batch2_data = np.asarray(batch2_data)
batch1_labels = np.asarray(batch1_labels)
batch2_labels = np.asarray(batch2_labels)
del x, y

print '---extra'

test = io.loadmat('./data/SVHN/format2/test_32x32.mat')
x = test['X'].transpose((2, 0, 1, 3)).reshape((3072, -1))
y = test['y'].reshape((-1,))
for i in np.arange(len(y)):
    if y[i] == 10:
        y[i] = 0
batch3_data = x
batch3_labels = []
for i in np.arange(len(y)):
    batch3_labels.append(y[i])
batch3_data = np.asarray(batch3_data).T
batch3_labels = np.asarray(batch3_labels)

print 'Check n x f'
print batch1_data.shape
print batch1_labels.shape
print batch2_data.shape
print batch2_labels.shape
print batch3_data.shape
print batch3_labels.shape

image = paramgraphics.mat_to_img(batch1_data[:100,:].T, dim_input, colorImg=colorImg, scale=True)
image.save(saveDir+'svhn_train.png', 'PNG')
image = paramgraphics.mat_to_img(batch2_data[:100,:].T, dim_input, colorImg=colorImg, scale=True)
image.save(saveDir+'svhn_valid.png', 'PNG')
image = paramgraphics.mat_to_img(batch3_data[:100,:].T, dim_input, colorImg=colorImg, scale=True)
image.save(saveDir+'svhn_test.png', 'PNG')

if preprocessing == 'gcn_var':
    batch1_data = pypp.global_contrast_normalize(batch1_data, subtract_mean=True, use_std=True)
    batch2_data = pypp.global_contrast_normalize(batch2_data, subtract_mean=True, use_std=True)
    batch3_data = pypp.global_contrast_normalize(batch3_data, subtract_mean=True, use_std=True)
elif preprocessing == 'gcn_norm':
    batch1_data = pypp.global_contrast_normalize(batch1_data, subtract_mean=True)
    batch2_data = pypp.global_contrast_normalize(batch2_data, subtract_mean=True)
    batch3_data = pypp.global_contrast_normalize(batch3_data, subtract_mean=True)
    print batch1_data.shape
    print batch1_data.max()
    print batch1_data.min()
    print batch2_data.shape
    print batch2_data.max()
    print batch2_data.min()
    print batch3_data.shape
    print batch3_data.max()
    print batch3_data.min()
else:
    print preprocessing
    print 'preprocessing unknown'
    exit()
image = paramgraphics.mat_to_img(batch1_data[:100,:].T, dim_input, colorImg=colorImg, scale=True)
image.save(saveDir+'svhn_train_'+preprocessing+'.png', 'PNG')
image = paramgraphics.mat_to_img(batch2_data[:100,:].T, dim_input, colorImg=colorImg, scale=True)
image.save(saveDir+'svhn_valid_'+preprocessing+'.png', 'PNG')
image = paramgraphics.mat_to_img(batch3_data[:100,:].T, dim_input, colorImg=colorImg, scale=True)
image.save(saveDir+'svhn_test_'+preprocessing+'.png', 'PNG')

f = file(saveDir+"svhn"+preprocessing+".bin","wb")
np.save(f,batch1_data)
np.save(f,batch1_labels)
np.save(f,batch2_data)
np.save(f,batch2_labels)
np.save(f,batch3_data)
np.save(f,batch3_labels)
f.close()