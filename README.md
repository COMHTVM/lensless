# cifar10_denoising
A tiny example application of pytorch_prototyping, with an example training script. The application is denoising of CIFAR10-images using a residual U-Net architecture. While the structure of this project is slightly overkill for the problem it's trying to solve, it is intended to serve as starter code for research projects. 

The training script and "DenoisingUnet" class handle:
1. checkpointing
2. Logging with tensorboardx
3. Writing evaluation results to disk

## Usage
### Installation
This code was developed in python 3.7 and Pytorch 1.0 (defaults to running on GPU). I recommend using anaconda for dependency management. 
You can create an environment with name "pytorch_starter" and all the dependencies necessary to run this code like so:
```
conda env create -f src/environment.yml
```
