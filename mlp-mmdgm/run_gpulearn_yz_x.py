'''
modified by Chongxuan Li (chongxuanli1991@gmail.com)
'''
import gpulearn_yz_x
import sys, os

if sys.argv[1] == 'svhn':
    n_hidden = [500,500]
    if len(sys.argv) == 4:
        n_hidden = [int(sys.argv[2])]*int(sys.argv[3])
    gpulearn_yz_x.main(dataset='svhn', n_z=300, n_hidden=n_hidden, seed=0, gfx=True)

elif sys.argv[1] == 'mnist':
    n_hidden = (500,500)
    if len(sys.argv) >= 4:
        n_hidden = [int(sys.argv[2])]*int(sys.argv[3])
    
    if os.environ.has_key('nz'):
        n_z = int(os.environ['nz'])
    else:
        n_z = 50
        
    if len(sys.argv) >= 5:
        n_z = int(sys.argv[4])
    gpulearn_yz_x.main(dataset='mnist', n_z=n_z, n_hidden=n_hidden, seed=0, gfx=True)
else:
    n_hidden = (500,500)
    n_z = 50
    gpulearn_yz_x.main(dataset=sys.argv[1], n_z=n_z, n_hidden=n_hidden, seed=0, gfx=True)
    #raise Exception("Unknown dataset")
