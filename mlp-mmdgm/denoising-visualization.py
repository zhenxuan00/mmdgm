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

f = gzip.open('data/mnist/mnist_28.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()

dir = sys.argv[1]

v = ndict.loadz(dir+'v_best.ndict.tar.gz')
w = ndict.loadz(dir+'w_best.ndict.tar.gz')

# choose number of images to transform and number of images to do visualization
num_trans = 1000
num_show = 300
data = (x_test[:num_trans,:]).T
pertub_label = np.ones(data.shape)

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
    _str = str(pertub_type)+'_params_'+str(int(pertub_prob*100))+'_'
    zz = sio.loadmat('data_imputation/type_'+str(pertub_type)+'_params_'+str(int(pertub_prob*100))+'_noise_rawdata.mat')
else:
    pertub_prob = int(pertub_prob)
    _str = str(pertub_type)+'_params_'+str(pertub_prob)+'_'
    zz = sio.loadmat('data_imputation/type_'+str(pertub_type)+'_params_'+str(pertub_prob)+'_noise_rawdata.mat')
data_perturbed = zz['z_test'][:,:num_trans]
pertub_label = zz['pertub_label'][:,:num_trans]
    
# denoise
print 'Denoising...'
n_hidden_q = 2
n_hidden_p = 2


output = np.zeros(data.shape+(denoise_times+2,))
output[:,:,0] = data
output[:,:,1] = data_perturbed

z_train = x_train.copy().T
#z_train = np.ones((v['w'+str(n_hidden_q-1)].shape[0],num_trans))

for t in xrange(2,denoise_times+2):
    tmp = output[:,:,t-1]
    # sample z
    for i in range(n_hidden_q):
        tmp = np.log(1 + np.exp(v['w'+str(i)].dot(tmp) + v['b'+str(i)]))
        # save features for prediction
        if t == 2:
            z_train = np.log(1 + np.exp(v['w'+str(i)].dot(z_train) + v['b'+str(i)]))
            sio.savemat(dir+_str+'noise_features.mat', {'z_train' : z_train, 'z_test' : tmp})
        if t == denoise_times+1 or (t>1 and (t-1)%100==0):
            sio.savemat(dir+_str+str(t-1)+'_de-noise_features.mat', {'z_train' : z_train, 'z_test' : tmp})
            
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
    
    output[:,:,t] = pertub_label*data+(1-pertub_label)*tmp

if pertub_type == 3:
    output[:,:,1] = pertub_label*data # ignore the random guess for visualization
sio.savemat(dir+_str+'de-noise_rawdata.mat', {'z_train' : x_train.T, 'z_test' : output[:,:,-1]})

# save data to do visualization
print 'Visualizing...'
visualization_image_number = num_show

# left a gap for sub-images
w = width+1
h = height+1

# layout the sub-images on a big image
w_image_number = int(math.sqrt(visualization_image_number) + 1)
h_image_number = int((visualization_image_number + w_image_number - 1) / w_image_number)
W = w * (w_image_number-1)+width
H = h * (h_image_number-1)+height
image = np.ones((H,W, denoise_times+2))

for t in xrange(denoise_times+2):
    for hn in xrange(h_image_number):
        for wn in xrange(w_image_number):
            #List the sub-images in rows 
            index = hn * w_image_number + wn
            if index < visualization_image_number:
                #May need transpose for w=h, but may have bugs for w ~= h
                image[hn*h:hn*h+height, wn*w:wn*w+width, t] = (output[:,index,t]).reshape((height, width))
                
sio.savemat('results/'+_str+'de-noise_visualization.mat', dict(image=image))

visualization_image_number = denoise_times+2
# layout the sub-images on a big image
w_image_number = int(math.sqrt(visualization_image_number) + 1)
h_image_number = int((visualization_image_number + w_image_number - 1) / w_image_number)
W = w * (w_image_number-1)+width
H = h * (h_image_number-1)+height
image1 = np.ones((H,W, num_show))

for n in xrange(num_show):
    for hn in xrange(h_image_number):
        for wn in xrange(w_image_number):
            #List the sub-images in rows 
            index = hn * w_image_number + wn
            if index < visualization_image_number:
                #May need transpose for w=h, but may have bugs for w ~= h
                image1[hn*h:hn*h+height, wn*w:wn*w+width, n] = (output[:,n,index]).reshape((height, width))
                
sio.savemat('results/'+_str+'de-noise_visualization1.mat', dict(image1=image1))


visualization_image_number = denoise_times+2
# layout the sub-images on a big image
w_image_number = visualization_image_number
h_image_number = 1
W = w * (w_image_number-1)+width
H = h * (h_image_number-1)+height
image2 = np.ones((H,W, num_show))

for n in xrange(num_show):
    for hn in xrange(h_image_number):
        for wn in xrange(w_image_number):
            #List the sub-images in rows 
            index = hn * w_image_number + wn
            if index < visualization_image_number:
                #May need transpose for w=h, but may have bugs for w ~= h
                image2[hn*h:hn*h+height, wn*w:wn*w+width, n] = (output[:,n,index]).reshape((height, width))
                
sio.savemat('results/'+_str+'de-noise_visualization2.mat', dict(image2=image2))

