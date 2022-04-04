# Non-params-2layers-ResNet

Follow Dr. Shay Cohen's paper Nonparametric Learning of Two-Layer ReLU Residual Units
https://arxiv.org/abs/2008.07648

PreReLU-TLRN solver using QP/LP (requires `cvx` Matlab software to run)

Before running, zoom in `cvx/` folder and run `cvx_setup` to set up the cvx.

# Files distributions:




  For synthetic experiments:
    The scripts are:
      `script_warmup.m`
      `script_synthetic1.m`
      `script_synthetic2.m`
      `script_synthetic2heatmap.m`

  For regression (WineQuality datasets):
  
    The dataset is in `winequality/` folder;
    
    The experiments scripts are:
      `script_wine_q_regression.m`, 
      `loadwinequality.m`

  For classification (Wine Datasets):
  
    The dataset is in `winedatasets/` folder
    
    The scripts are:
      `script_small_classify.m`, 
      `loadwinedata.m`

  For classification (CIFAR-10):
  
    The dataset is in `cifar-10-bathces-mat/` folder;
    Use script `storecifar10.m` to collect data into cifar10data.mat
    The experiments scripts are:
      `get_cifar10_data.m`, 
      `script_cifar10.m`, 
      `script2_cifar10.m`

   Some helper functions:
   
    `generatenoise.m`, 
    `mymse.m`, 
    `PadY.m`
  
    
