'''
Code for mmDGM
Author: Chongxuan Li (chongxuanli1991@gmail.com)
Version = '1.0'
'''

import numpy as np
from anglepy import ndict
import scipy.io as sio
import cPickle, gzip
import math
import os, sys

# load data, recognition model and generative model
print 'Loading data...'

dir = sys.argv[1]
p_type = sys.argv[5]
if p_type == 'null':
    p_type = ''

v = ndict.loadz(dir+'v'+p_type+'.ndict.tar.gz')
w = ndict.loadz(dir+'w'+p_type+'.ndict.tar.gz')

# perturb data
print 'Loading perturbed data...'

width = 28
height = 28
denoise_tpye = 1 # sample or mean 
pertub_type = int(sys.argv[2])
pertub_prob = float(sys.argv[3])
denoise_times = int(sys.argv[4]) # denoising epoch

print pertub_type, pertub_prob, denoise_times

if pertub_type == 4:
    _str = str(pertub_type)+'_params_'+str(int(pertub_prob*100))+p_type+'_mse'
    zz = sio.loadmat('data_imputation/type_'+str(pertub_type)+'_params_'+str(int(pertub_prob*100))+'_noise_rawdata.mat')
else:
    pertub_prob = int(pertub_prob)
    _str = str(pertub_type)+'_params_'+str(pertub_prob)+p_type+'_mse'
    zz = sio.loadmat('data_imputation/type_'+str(pertub_type)+'_params_'+str(pertub_prob)+'_noise_rawdata.mat')
#fw = open(dir+_str,'w')
    
data = zz['z_test_original']
data_perturbed = zz['z_test']
pertub_label = zz['pertub_label']
pertub_number = float(np.sum(1-pertub_label))

start_mse = float(np.sum((data_perturbed-data)**2))/pertub_number

#print pertub_number, data.shape[0]*data.shape[1], start_mse, 
#print data.dtype, data_perturbed.dtype, pertub_label.dtype
#print '---'
    
# denoise
print 'Denoising...'
n_hidden_q = 2
n_hidden_p = 2

output = data_perturbed.copy()
tmp = output.copy()

for t in xrange(denoise_times):
    tmp = output
    # sample z
    for i in range(n_hidden_q):
        tmp = np.log(1 + np.exp(v['w'+str(i)].dot(tmp) + v['b'+str(i)]))
        
    q_mean = v['mean_w'].dot(tmp) + v['mean_b']
    
    if denoise_tpye == 1:
        q_logvar = v['logvar_w'].dot(tmp) + v['logvar_b']
        eps = np.random.normal(0, 1, (q_mean.shape))
        tmp = q_mean + np.exp(0.5*q_logvar) * eps
    elif denoise_tpye == 2:
        tmp = q_mean
        
    # generate x
    for i in range(n_hidden_p):
        tmp = np.log(1 + np.exp(w['w'+str(i)].dot(tmp) + w['b'+str(i)]))
    tmp = 1/(1 + np.exp(-(w['out_w'].dot(tmp)+w['out_b'])))
    
    output = pertub_label*data+(1-pertub_label)*tmp
    
    # compute MSE between output and data
    mse1 = float(np.sum((output-data)**2))/pertub_number
    #print '*', np.sum(pertub_label*((output-data)**2))
    #print '*', np.mean((1-pertub_label)*((output-data)**2))
    mse2 = np.mean((output-data)**2)
    print mse1
    #fw.writelines(str(t)+' ')
    #fw.writelines(str(mse1)+' ')
    #fw.writelines(str(mse2)+'\n')

