Code for NIPS 2015 paper on Max-margin Deep Generative Models(MMDGM)

Chongxuan Li, Jun Zhu, Tianlin Shi and Bo Zhang,
Max-margin Deep Generative Models 
Advances in Neural Information Processing Systems (NIPS15), Montreal  
Please cite this paper when using this code for your research.

For questions and bug reports, please send me an e-mail at _chongxuanli1991[at]gmail.com_.


## Prerequisites

1. Some libs we used in our experiments:
    - Python (version 2.7)
	- Numpy
    - Scipy
    - Theano (version 0.7.0)
    - Cuda (7.0)
    - Cudnn (optional, see details below, version 6.5)
    - Pylearn2 (for pre-processing of SVHN)

2. GPU: TITAN X Black
    - memory of GPU is at least 9G for SVHN

## Some matters

1. We found that theano is computationally unstable with Cudnn. For MNIST, we do NOT use Cudnn and the error rate should be exact 0.45% with same version of libs and machine. For SVHN, we DO use Cudnn for faster training and there exists additional randomness even given fixed random seed. However, we run this experiments for 5 times and choose the lowest accuracy to report. Typically, the generative results won't change much given different version of libs. 

2. In MLP case, we use code from Kingma. In CNN case, we use code from Goodfellow to do local contrast normalization(LCN) for SVHN data(pylearn2). We also use code from the tutorial on deeplearning.org.

3. I didn't upload our trained model to github and you may need to train models following the command below and put models in the right place. For a full version of code with trained models, you can access http://ml.cs.tsinghua.edu.cn/~chongxuan/static/mmdgm_release.rar.

## MLP results

```
# We did our experiments based on Kingma's code[https://github.com/dpkingma/nips14-ssl]. 
Export data path: 
export ML_DATA_PATH="[dir]/mmdgm/mlp_mmdgm/data"

# VA on MNIST with pre-training, lower bound: 
run_va.sh
# Train the model without pre-training by setting the [pretrain] flag to 0 in the .sh file.

# VA on MNIST, error rate:
python pegasos.py dir/full_latent.mat mnist

# MMVA on MNIST with pre-training:
run_mmva.sh

# MSE results with missing value on MNIST: 
    - mse_va: python mse_denoising.py models/va_3000/ 3 12 100 _best
    - mse_mmva: python mse_denoising.py models/mmva_3000/ 3 12 100 _best
    - visualization: python denoising-visualization.py models/mmva_3000/ 3 12 25
    Generate data by yourself:
        - rectangle: python generate_data_mnist.py 3 12 (size of rectangle, an even number less than 28)
        - random drop: python generate_data_mnist.py 4 0.8 (drop ratio, a real number in range (0, 1))
        - half: python generate_data_mnist.py 5 0 14 (integer less than 28)
```

## CNN results on MNIST

```
# CVA on MNIST, lower bound: 
run_supervised_6layer_cva_mnist.sh

# CVA on MNIST, error rate: 
run_supervised_cva_svm_mnist.sh

# CMMVA on MNIST with default value of C: 
run_supervised_6layer_cmmva_mnist.sh
# You could set D=1,1e-1,1e-2,1e-4 to obtain Table2 in the paper.

# MSE results with missing value on MNIST:
    - generate data at first: python generate_data_mnist.py 3 12
    - cva: run_imputation_mse_cva_mnist.sh
    - cmmva: run_imputation_mse_cmmva_mnist.sh
    Generate other types of data by yourself:
        - rectangle: python generate_data_mnist.py 3 12 (size of rectangle, an even number less than 28)
        - random drop: python generate_data_mnist.py 4 0.8 (drop ratio, a real number in range (0, 1))
        - half: python generate_data_mnist.py 5 0 14 (integer less than 28)

# Classification results with missing value on MNIST:
    - cnn: run_imputation_classification_cmm_mnist.sh
    - cva: run_imputation_classification_cva_mnist.sh
    - cmmva: run_imputation_classification_cmmva_mnist.sh
    Train cnn model: run_supervised_6layer_cnn_mnist.sh.
```

## CNN results on SVHN

```
# The data is too large to upload. Firstly, download the online dataset in .mat format and run preprocessing_svhn.sh to preprocess the data. (This pre-processing procedure should be done WITHOUT Cudnn to obtain a stable version of data.)

# CVA on SVHN, lower bound: 
run_supervised_6layer_cva_svhn.sh

# CVA on SVHN, error rate: 
run_supervised_cva_svm_svhn.sh

# CMMVA on SVHN with default value of C: 
run_supervised_6layer_cmmva_svhn.sh.
# We pre-trained our recognition model separately without dropout 10 epochs for fast convergence. Pre-train this model by yourself: 
run_supervised_6layer_cnn_svhn.sh.

# Missing value imputation: 
    - cva：run_imputation_cva_svhn.sh
    - cmmva: run_imputation_cmmva_svhn.sh
    Generate data by yourself:
        python generate_data_svhn_1000_for_test.py 3 12
```




