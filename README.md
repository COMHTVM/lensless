# End-to-end optimization of a lensless imaging system

Final project for Winter 2020 iteration of EE 367: Computational Imaging at Stanford.

This repo contains code to perform end-to-end optimization of a plastic phase mask placed close to the sensor (< 25 mm focal distance).

## Pipeline
![image](link)

We implement an optics module, sensor module, and U-Net for image reconstruction. The loss is backpropagated into the heightmap to optimize a coded phase mask.

## File guide
* `train.py` is the main training script
* `params.json` is used to load hyperparameters relevant for training and initialization
* `ranges.json` is used for queueing up multiple experiments (which will stop depending on early stopping criteria)
* `dataio.py` contains the dataloader that will load the SBDataset for training
* `denoising_unet.py` contains the model and model helper functions

## Running code
* Edit `params.json` and `ranges.json` as needed for experiment
    * If first time running, set `download_data` to be `true`. After first run, set this to `false`.
* In console, run `CUDA_VISIBLE_DEVICES=# python3 train.py` where `#` specifies GPU device number in the project folder.
* Data generated from the experiment (saved models and Tensorboard files) will be specified in `runs/exp_name/exp_name_#` where `exp_name` is as specified in hyperaparameters and `#` is automatically determined. 

## Dependencies
* pytorch
* numpy
* tensorboard
* [U-Net repo](https://github.com/vsitzmann/cifar10_denoising)
* [Propagation and utils repo](https://github.com/computational-imaging/citorch)