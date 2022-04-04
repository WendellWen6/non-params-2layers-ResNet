# Non-params-2layers-ResNet

Follow Dr. Shay Cohen's paper Nonparametric Learning of Two-Layer ReLU Residual Units
https://arxiv.org/abs/2008.07648

PreReLU-TLRN solver using QP/LP (requires `cvx` Matlab software to run)

Before running, zoom in `cvx/` folder and run `cvx_setup` to set up the cvx.

# Files distributions:

  For synthetic experiments:
  
    script_warmup.m  -- run the warmup exp
    script_synthetic1.m -- run with different Y padding methods
    script_synthetic2.m -- generate data for heatmap
    script_synthetic2heatmap.m -- generate heatmap

  For regression (WineQuality datasets):
  
    The dataset is in winequality/ folder;
    
    script_regression.m
    loadwinequality.m

  For classification (Wine Datasets):
  
    The dataset is in winedatasets/ folder

    script_classification.m
    loadwinedata.m

  For classification (CIFAR-10):
  
    The dataset is in cifar-10-bathces-mat/ folder;
    Use script storecifar10.m to collect data into cifar10data.mat
    
    get_cifar10_data.m
    script_cifar10.m -- exp with d and n 
    script2_cifar10.m -- heatmap

   Some helper functions:
   
    generatenoise.m -- use our model to generate data to be used as noise
    mymse.m -- my mean square error functino
    PadY.m -- padding Y methods
    calculate_acc.m -- for cifar10 datasets
    calculate_error_acc.m -- for wine datasets
    
  The rest of the codes are the core codes of model from Dr. Shay Cohen's projects
    
    
    
  
    
