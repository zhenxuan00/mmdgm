import numpy as np
import scipy.io as sio
import cPickle, gzip
import math
import os, sys
from util import paramgraphics

# load data
print 'Loading data...'

f = gzip.open('./data/mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()

# choose number of images to transform
num_trans = 10000
data = (x_test[:num_trans,:]).T
pertub_label = np.ones(data.shape)

# perturb data
print 'Perturbing data...'
width = 28
height = 28

pertub_type = int(sys.argv[1])
pertub_prob = float(sys.argv[2])
noise_type = 2 # 0 or uniformly random

if pertub_type == 1:
    data_perturbed = data + np.random.normal(0,0.4,(data.shape))
elif pertub_type == 2:
    data_perturbed = data.copy()
    data_perturbed *= (np.random.random(data.shape) > pertub_prob)
elif pertub_type == 3:
    data_perturbed = data.copy()
    pertub_prob = int(pertub_prob)
    rec_h = pertub_prob
    rec_w = rec_h
    begin_h = (width - rec_w)/ 2
    begin_w = (width - rec_w)/ 2
    
    print rec_h, rec_w, begin_h, begin_w
    rectengle = np.zeros(rec_h*rec_w)
    
    for i in xrange(rec_h):
        rectengle[i*rec_w:(i+1)*rec_w]=np.arange((begin_h+i)*width+begin_w,(begin_h+i)*width+begin_w+rec_w)
    if noise_type == 1:
        data_perturbed[rectengle.astype(np.int32),:] = 0
    else:
        data_perturbed[rectengle.astype(np.int32),:] = np.random.random((rectengle.shape[0],data.shape[1]))
    
    pertub_label[rectengle.astype(np.int32),:] = 0
elif pertub_type == 4:
    data_perturbed = np.random.random(data.shape)
    sample = np.random.random(data.shape)
    pertub_label[sample < pertub_prob] = 0
    data_perturbed = pertub_label*data+(1-pertub_label)*data_perturbed

elif pertub_type == 5:
    pertub_prob1 = float(sys.argv[3])
    start = int(pertub_prob)
    end = int(pertub_prob1)
    data_perturbed = np.zeros(data.shape)
    tmp_a = np.ones(width)
    tmp_a[start:end] = 0
    #print tmp_a.shape
    #print tmp_a
    tmp_b = np.tile(tmp_a, height)
    print tmp_b.shape
    print pertub_label.shape
    pertub_label = (pertub_label.T*tmp_b).T
    data_perturbed = pertub_label*data+(1-pertub_label)*data_perturbed
    
    
if pertub_type == 4:
    sio.savemat('data_imputation/type_'+str(pertub_type)+'_params_'+str(int(pertub_prob*100))+'_noise_rawdata.mat', {'z_train' : x_train.T, 'z_test_original' : data, 'z_test' : data_perturbed, 'pertub_label' : pertub_label})
    #print data_perturbed[:,:25].shape
    image = paramgraphics.mat_to_img(data_perturbed[:,:25], (28,28), colorImg=False, scale=True)
    image.save('data_imputation/test_noise_4_'+str(pertub_prob)+'.png', 'PNG')
elif pertub_type == 3:
    sio.savemat('data_imputation/type_'+str(pertub_type)+'_params_'+str(pertub_prob)+'_noise_rawdata.mat', {'z_train' : x_train.T, 'z_test_original' : data, 'z_test' : data_perturbed, 'pertub_label' : pertub_label})
    #print data_perturbed[:,:25].shape
    image = paramgraphics.mat_to_img(data_perturbed[:,:25], (28,28), colorImg=False, scale=True)
    image.save('data_imputation/test_noise_3_'+str(pertub_prob)+'.png', 'PNG')

elif pertub_type == 5:
    sio.savemat('data_imputation/type_'+str(pertub_type)+'_params_'+str(start)+'_'+str(end)+'_noise_rawdata.mat', {'z_train' : x_train.T, 'z_test_original' : data, 'z_test' : data_perturbed, 'pertub_label' : pertub_label})
    #print data_perturbed[:,:25].shape
    image = paramgraphics.mat_to_img(data_perturbed[:,:25], (28,28), colorImg=False, scale=True)
    image.save('data_imputation/test_noise_5_'+str(start)+'_'+str(end)+'.png', 'PNG')
