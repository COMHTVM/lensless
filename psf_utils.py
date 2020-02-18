import numpy as np
from propagation import Propagation
import torch
import math
import sys
import time

sys.path.append("../")
from utils import *

class DOEOpt:
    def __init__(self, kernel, distance, slm_resolution, pixel_pitch, wavelength=532e-9):
        # distance and pixel_pitch in meters
        self.prop = Propagation(kernel_type=kernel, propagation_distances=distance,
                                slm_resolution=slm_resolution,
                                slm_pixel_pitch=pixel_pitch,
                                wavelength=wavelength)

def circular_aperture(input_field, r_cutoff):
    ''' Cuts off values outside of aperture area
    Parameters:
        input_field: tensor/array of incoming light field
            (height, width, 2) - last dimension for real and imaginary
        r_cutoff: int in pixels of radius of aperture
    Output:
        filtered: Torch tensor of outgoing light field
    '''
    input_field = input_field.numpy()
    input_shape = input_field.shape
    [x,y] = np.mgrid[-input_shape[0]//2 : input_shape[0]//2, -input_shape[1]//2:input_shape[1]//2].astype(np.float64)
    if r_cutoff is None:
        r_cutoff = np.amax(x)
    r = np.sqrt(x**2 + y**2)
    aperture = (r < r_cutoff)
    real = input_field[:,:,0]*aperture
    imag = input_field[:,:,1]*aperture
    filtered = np.stack((real,imag),axis=-1)
    return torch.Tensor(filtered)

# Models a simple lens
def plano_convex_initializer(focal_length, resolution, pixel_pitch, wave_length, refractive_idc):
    convex_radius = (refractive_idc - 1.) * focal_length # based on lens maker formula

    N = resolution
    M = resolution
    [x, y] = np.mgrid[-N // 2:N // 2, -M // 2:M // 2].astype(np.float64)

    x = x * pixel_pitch
    y = y * pixel_pitch

    # get lens thickness by paraxial approximations
    height_map = -(x ** 2 + y ** 2) / 2. * (1. / convex_radius)

    return torch.Tensor(height_map)

def get_psf(hyps):
    resolution = [hyps['resolution'],hyps['resolution']]
    physical_size = hyps['pixel_pitch']*hyps['resolution']

    N,M = resolution
    [x, y] = np.mgrid[-N // 2:N // 2, -M // 2:M // 2].astype(np.float64)
    x = x / N * physical_size
    y = y / M * physical_size
    squared_sum = torch.Tensor(x ** 2 + y **2)
    wave_nos = 2. * math.pi / hyps['wavelength']

    curvature = torch.sqrt(squared_sum + hyps['distance'])
    height_map = plano_convex_initializer(hyps['focal_length'],
                                          hyps['resolution'],
                                          hyps['pixel_pitch'],
                                          hyps['wavelength'],
                                          hyps['refractive_idx'])

    phase_delay = heightmap_to_phase(height_map, hyps['wavelength'],
                                     hyps['refractive_idx'])
    spherical = polar_to_rect(1, wave_nos * curvature)
    spherical = stack_complex(spherical[0], spherical[1])
    phases = polar_to_rect(1, phase_delay)
    phases = stack_complex(phases[0], phases[1])
    realimag = mul_complex(spherical, phases)

    if hyps['r_cutoff'] is not None:
        final_r = circular_aperture(realimag, hyps['r_cutoff'])
    else:
        final_r = realimag

    pixel_pitch = [hyps['pixel_pitch'], hyps['pixel_pitch']]

    doe = DOEOpt(kernel='fresnel',
                 distance=hyps['focal_length'],
                 slm_resolution=resolution,
                 pixel_pitch=pixel_pitch,
                 wavelength=hyps['wavelength'])
    field = doe.prop.forward(final_r)
    intensity = field_to_intensity(field)
    return intensity
