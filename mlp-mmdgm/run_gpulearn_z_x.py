'''
modified by Chongxuan Li (chongxuanli1991@gmail.com)
'''
import gpulearn_z_x
import sys, os
import time
import color

n_hidden = (500,500)
if len(sys.argv) > 2:
  n_hidden = tuple([int(x) for x in sys.argv[2:]])
nz=500
if os.environ.has_key('nz'):
  nz = int(os.environ['nz'])
if os.environ.has_key('random_seed'):
    seed = 0
    if bool(int(os.environ['random_seed'])):
        seed = int(time.time())
    color.printRed('random_seed ' + str(seed))
else:
    seed = int(time.time())
    color.printRed('random_seed ' + str(seed))
gpulearn_z_x.main(dataset=sys.argv[1], n_z=nz, n_hidden=n_hidden, seed=seed, comment='', gfx=True)

