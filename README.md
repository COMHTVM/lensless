# End-to-end optimization of a lensless imaging system

Final project for Winter 2020 iteration of EE 367: Computational Imaging at Stanford.

Author: Cindy Nguyen, cindyn at stanford.edu

This repo contains code to perform end-to-end optimization of a plastic phase mask placed close to the sensor (<= 25 mm focal distance).

## Pipeline
![image](link)

We implement an optics module, sensor module, and Wiener deconvolution for image reconstruction. The loss is backpropagated into the heightmap to optimize a coded phase mask.

## File guide
* `train.py` is the main training script (implements sequence of experiments as a queue)
* `params.json` is used to load hyperparameters relevant for training and initialization (see `README.md` for details)
* `ranges.json` is used for queueing up multiple experiments (which will stop depending on early stopping criteria)
* `dataio.py` contains the dataloader that will load the SBDataset for training
* `denoising_unet.py` contains the model and model helper functions
* `notebooks/analyze_models.ipynb` loads models and plots losses

## Running code
* `git clone` this repository
* Edit `params.json` and `ranges.json` as needed for experiment (see ```PARAMS.md``` for details)
    * If running script to test is things are set up properly, you can run with the settings as is. This will load in a single image and beginning optimizing a height map and damping factor starting from an in-focus Fresnel lens height map.
    * If you want to run with the dataset, set the parameter `download_data` to be `true`. After first run, set this to `false`.
* In console, run 
```ssh
$ CUDA_VISIBLE_DEVICES=# python3 train.py
```
where `#` specifies GPU device number.
* Data generated from the experiment (saved models and Tensorboard files) will be specified in `runs/exp_name/exp_name_#` where `exp_name` is as specified in hyperaparameters and `#` is automatically determined.
* The training script is set up so that a new experiment is created for each hyperparameter in `ranges.json`, run each sequentially to completion, and save model checkpoints during training in the `runs` folder. 
* Data files are not included to save space.

## Dependencies
* pytorch
* numpy
* tensorboard
* [U-Net repo](https://github.com/vsitzmann/cifar10_denoising)
* [Propagation and utils repo](https://github.com/computational-imaging/citorch)
