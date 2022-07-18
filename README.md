# Tensor-based Multi-view Spectral Clustering via Shared Latent Space

## Abstract

Multi-view Spectral Clustering (MvSC) attracts increasing attention  due to nowadays  diverse data sources. However, existing works mainly focus on accuracy, whereas  model interpretability  and  clustering results are interesting to explore and yet are overlooked. Besides, most of these methods are inapplicable to  out-of-sample predictions  after training.
In this paper, a new method for MvSC is proposed via a shared latent space derived from the Restricted Kernel Machine (RKM) framework. Through the lens of conjugate feature duality in RKM, we cast the corresponding weighted kernel principal component analysis problem for MvSC, and develop a modified weighted conjugate feature duality  to 
formulate the dual variables. 
In the proposed method, the dual variables, which play the role of  hidden features, are imposed to be shared by all views to construct a common latent space, which couples all views by learning projections from the  spaces of 
different views.
Such single latent space promotes well-separated  clusters and provides straightforward  data exploration in terms of the dual variables, 
facilitating data visualization and interpretation.
In the resulting optimization, our method requires only the solution of a single eigenvalue decomposition
problem, whose computational complexity is independent of the number of views. To boost higher-order correlations,  tensor-based representations are introduced to the  modelling without increasing computational complexity.
The proposed method can be flexibly applied with out-of-sample extensions,  enabling greatly improved efficiency for large-scale data with fixed-size kernel schemes.
Numerical experiments verify that our method is effective in terms of accuracy, efficiency, and interpretability, showing a sharp eigenvalue decay and distinct latent variable distributions.

## Code Structure

- Training and evaluation of a TMvKSCR model is done in `mvkscrkm.py` from the `main`.
  - All cluster assignment methods (`uncoupled` and `mean`) are always evaluated.
- Comparison with KSC and MV-KSC is done in `ksc.py` and `mvksclssvm.py`, respectively.
- Available kernels are in `kernels.py`. 
  - To add a new kernel, define a new pytorch module, similar to, e.g., `GaussianKernelTorch`. Then, add that kernel to the `kernel_factory` function.
- Employed datasets are in `dataloader.py`.
  - To add a new dataset, define a new `get_dataset_dataloader` function returning the pytorch dataloader. Then, add that dataset to the `get_dataloader_helper` function.

## Usage
### Download
First, navigate to the unzipped directory and install required python packages with the provided `requirements.txt` file. This is explained in the following section.

### Install packages in conda environment
Run the following in terminal. This will create a conda environment named *rkm_env*.

```
conda create --name rkm_env python=3.8
```

Activate the conda environment with the command `conda activate rkm_env`. To install the required dependencies, run:

```R
pip install -r requirements.txt
```

### Train

Activate the conda environment `conda activate rkm_env` and run one of the following commands, for example:
```
python mvkscrkm.py dataset=toydataset2 kernels=toydataset2_rbf
python mvkscrkm.py dataset=data3Sources kernels=data3Sources_normpoly
python mvkscrkm.py dataset=ads kernels=ads_rbf
```

The configuration is done using YAML files with [hydra](https://hydra.cc/). Available configurations are in the `configs` directory. You may edit the provided YAML files or write your own files. Alternatively, you can change configurations directly from the command line, e.g,.:

```
python mvkscrkm.py model.eta=3.0
```

The following options for training a TMvKSCR model are available:

```
$ python mvkscrkm.py --help
mvkscrkm is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

dataset: ads, data3Sources, game, kolenda, nus, reuters, reuters2, toydataset2, toydataset3
kernels: ads_rbf, ads_rbf_single, data3Sources_normpoly, data3Sources_normpoly_single, kolenda_normpoly, linear_single, normpoly_5views, rbf_3views, rbf_5views, rbf_6views, rbf_single, reuters2_normpoly, reuters2_normpoly_single, toydataset2_rbf, toydataset2_rbf_single, toydataset3_rbf, toydataset3_rbf_single
model: ksc, lssvm, rkm


== Config ==
Override anything in the config (foo.bar=value)

dataset:
  name: toydataset2
  'N': -1
  mb_size: 1
  Ntest: 0
  shuffle: false
  workers: 0
  k: 2
  k_folds: 0
  normalize: false
model:
  name: rkm
  eta: 1.0
  assignment:
    beta:
      value: null
      beta1: null
      beta2: null
      beta3: null
      beta4: null
      beta5: null
      beta6: null
      beta7: null
      beta8: null
      beta9: null
      beta10: null
  rho: 1.0
  kappa:
    value: 1.0
    kappa1: null
    kappa2: null
    kappa3: null
    kappa4: null
    kappa5: null
    kappa6: null
    kappa7: null
    kappa8: null
    kappa9: null
    kappa10: null
kernels:
  kernel1:
    name: rbf
    args:
      sigma2: 3
  kernel2:
    name: rbf
    args:
      sigma2: 3
  kernel3:
    name: rbf
    args:
      sigma2: 3
seed: 0


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```
