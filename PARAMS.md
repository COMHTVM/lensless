# Guide to hyperparameters

* `data_root`: folder containing image dataset
* `logging_root`: folder to save all experiments in
* `train_test`: specify `train` or `val` to specify which dataset to load
* `exp_name`: name of experiment to specify
* `checkpoint`: load checkpoint file if you want to continue training from a previous training session
* `max_epoch`: maximum number of epochs
* `lr`: learning rate
* `scheduler`: type of scheduler (not implemented yet)
* `batch_size`: batch size
* `reg_weight`: regularizer factor
* `K`: a fixed damping factor for Wiener filtering
* `learn_weiener`: `true` or `false` to learn wiener filtering damping factor K
* `resolution`: int for resolution of point spread function
* `pixel_pitch`: float for pixel size
* `focal_length`: float for distance between phase mask and sensor
* `r_cutoff`: `null` or an int for the radius of the aperture
* `refractive_idc`: refractive index of phase mask
* `wavelength`: float for wavelength of interest
* `init_lens`: `random`, `fresnel`, `plano` for initializing the height map of the phase mask
* `single_image`: boolean for testing optimizing phase mask over one image
* `download_data`: boolean for downloading data (should be `false` after first download)