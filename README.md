# _Global Convergence and Induced Kernels of Gradient-Based Meta-Learning with Deep Neural Nets_

This repository contains an official implementation for the paper, _Global Convergence and Induced Kernels of Gradient-Based Meta-Learning with Deep Neural Nets_. In the paper, we theoretically derive that Model-Agnostic Meta-Learning (MAML) with infinitely wide neural networks is equivalent to a special kernel regression with a new class of kernels, which we name s Meta Neural Kernels (MNKs). The MNK can be used as a kernel-based meta-learning method. We compare MNK vs. MAML and implicit MAML (iMAML) on a popular few-shot image classification benchmark, the Omniglot dataset, and find MNK can outperform MAML and iMAML in the _small-data_ cases. This repository includes our implementation of MNK and the code to compare it with MAML & iMAML.

If you find this repository useful for your research, please consider citing our work:

`@article{meta-learning-convergence-kernel,
    title={Global Convergence and Induced Kernels of Gradient-Based Meta-Learning with Neural Nets},
    author = {Wang, Haoxiang and Sun, Ruoyu and Li, Bo},
    year={2020},
  	journal={To appear on arXiv},
}`


## Code
We provide a Jupyter notebook with comments to reproduce our experiments.
`Experiments.ipynb`: run the experiments for Meta Neural Kernels (MNK), MAML and iMAML on subsets of Omniglot. The code will automatically load pretrained models and dataset from `saved_models/`. 
`meta_cntk.py` defines the MNK model.
`saved_models` contains the preprocessed dataset and pre-trained models.

## To Download
Due to the large size of the dataset and pre-trained models, we provide a Dropbox folder containing them: [link](https://www.dropbox.com/sh/2us89h35i3r34zu/AAB6LpqavUoZc1vCKYe1Sw9Ua?dl=0). Please click the link and download the files into `saved_models/`.

**Note**: The computation and memory cost of kernel methods is quadratic in the number of samples. Hence, it is very expensive to compute the MNK values on Omniglot. For instance, our pre-trained model takes about 1000 GPU hours on NVIDIA RTX 2080ti. That is why we provide pre-computed kernel values. However, since the computation cost is quadratic, MNK is not that computationally expensive for small-data tasks.

## Dependency
We run experiments on Ubuntu 18.04 with Python 3.7 and CUDA 10.2. The following python packages are required.
numpy, scipy, matplotlib, pandas
pytorch=1.5
cupy
scikit-image
scikit-learn
higher
torchmeta

**Note**: the denpendency list shown above may not be complete. We will update it later.

## Open-Source Code Repositories
We adopt several open-sourced GitHub code repositories in this codebase, and we are grateful to their authors:
[CNTK](https://github.com/ruosongwang/CNTK)
[higher](https://github.com/facebookresearch/higher/)
[hypertorch](https://github.com/prolearner/hypertorch)



## To-Do
+ Provide configurations/hyperparmeters for reported experiments in the paper. Will be put into a new `configs/` folder.
+ Provide code that can compute MNK kernel values from raw data (e.g., any data in the similar form to Omniglot).
+ Provide a docker container that makes the experiments easier to run.


