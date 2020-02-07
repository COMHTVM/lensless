import numpy as np
from propagation import Propagation
import torch
import math
import matplotlib.pyplot as plt
import sys
import time

sys.path.append("../")
from utils import *

cuda = torch.device('cuda:0')
wave_lengths = np.array([635, 530, 450]) * 1e-9      # RGB wavelengths
refractive_idcs = np.array([1.4295, 1.4349, 1.4421]) # RGB idcs

class DOEOpt:
    def __init__(self, kernel, distance, slm_resolution, pixel_pitch, wavelength=532e-9):
        # distance and pixel_pitch in meters
        self.prop = Propagation(kernel_type=kernel, propagation_distances=distance,
                                slm_resolution=slm_resolution,
                                slm_pixel_pitch=pixel_pitch,
                                wavelength=wavelength)

def circular_aperture(input_field, r_cutoff):
    input_field = input_field.numpy()
    input_shape = input_field.shape
    [x,y] = np.mgrid[-input_shape[0]//2 : input_shape[0]//2, -input_shape[1]//2:input_shape[1]//2].astype(np.float64)
    if r_cutoff is None:
        r_cutoff = np.amax(x)
    r = np.sqrt(x**2 + y**2)
    aperture = (r < r_cutoff)
    real = input_field[:,:,0]*aperture
    imag = input_field[:,:,1]*aperture
    a = np.stack((real,imag),axis=-1)

    return torch.Tensor(a)

# Models a simple lens
def plano_convex_initializer(focal_length, resolution, pixel_pitch, wave_length, refractive_idc):
    convex_radius = (refractive_idc - 1.) * focal_length # based on lens maker formula

    N, M = resolution
    [x, y] = np.mgrid[-N // 2:N // 2, -M // 2:M // 2].astype(np.float64)

    x = x * pixel_pitch
    y = y * pixel_pitch

    # get lens thickness by paraxial approximations
    height_map = -(x ** 2 + y ** 2) / 2. * (1. / convex_radius)

    return torch.Tensor(height_map)

# Models a fresnel lens
def simple_to_fresnel_lens(phase_delay, resolution):

    N,M = resolution

    min_phase = phase_delay.min()

    phase_delay -= min_phase
    max_phase = phase_delay.max()

    fresnel = torch.zeros([N, M])

    threshold = 2 * math.pi

    count = 1
    while max_phase > threshold:
        maskA = phase_delay < threshold  # mask for indices lower than 2*pi
        maskB = 0 < phase_delay  # mask for indices greater than 0
        fullmask = maskA * maskB  # mask for indices 0 < x < 2*pi

        outer_circle = phase_delay * fullmask
        fresnel += outer_circle  # add to an empty 2d array to build fresnel lens

        phase_delay = phase_delay * np.invert(fullmask)  # acquire the unconverted lens
        phase_delay -= threshold  # subtract 2pi from unconverted lens
        phase_delay = torch.clamp(phase_delay,min=0)
        max_phase = phase_delay.max()
        count += 1
    fresnel += phase_delay
    fresnel -= threshold

    return torch.Tensor(fresnel)

def get_psfs(distances, focal_length, physical_size=3e-2, wave_length=532e-9,
             refractive_idc=1, lens_type='simple', prop_type='fresnel',
             r_cutoff=80):
    N, M = resolution
    [x, y] = np.mgrid[-N // 2:N // 2, -M // 2:M // 2].astype(np.float64)

    x = x / N * physical_size
    y = y / M * physical_size
    squared_sum = torch.Tensor(x ** 2 + y ** 2)

    wave_nos = 2. * math.pi / wave_length
    input_fields = []

    for distance in distances:

        # distance between point source and points on the incident plane
        curvature = torch.sqrt(squared_sum + distance ** 2)

        height_map = plano_convex_initializer(focal_length, resolution, pixel_pitch[0],
                                            wave_length, refractive_idc)

        # get the phase shifts based on thickness of the lens
        phase_delay = heightmap_to_phase(height_map, wave_length, refractive_idc)

        if lens_type == 'fresnel':
            phase_delay = simple_to_fresnel_lens(phase_delay, resolution)

        # propagate light through lens
        spherical = polar_to_rect(1, wave_nos * curvature)
        spherical = stack_complex(spherical[0],spherical[1])
        phases = polar_to_rect(1, phase_delay)
        phases = stack_complex(phases[0],phases[1])
        realimag = mul_complex(spherical, phases)

        # put light through aperture
        final_r = circular_aperture(realimag, r_cutoff)
        input_fields.append(final_r)

    psfs = []

    for depth_idx, input_field in enumerate(input_fields):
        doe = DOEOpt(kernel=prop_type, distance=focal_length, slm_resolution=resolution,
                     pixel_pitch=pixel_pitch,
                     wavelength=wave_length)

        field = doe.prop.forward(input_field) # free space propagation between lens and sensor
        intensity = field_to_intensity(field)
        psfs.append(intensity)
    return psfs

def plot_full_psf(psfs, num_plots,f):
    fig, axs = plt.subplots(num_plots, 1, figsize = (20, 30))
    for i in range(num_plots):
        im = axs[i].imshow(psfs[i], vmin=0, vmax=1)
        fig.colorbar(im, ax=axs[i])
    plt.show()

def plot_side_psf(r,g,b,num_plots):
    # r,g,b list of psfs each, for a 501x501 plot
    f = plt.figure(figsize=(4, 35))
    pos = 1
    for i in range(num_plots):
        f.add_subplot(10, 1, pos)
        slice_r = r[i][251, :]
        slice_g = g[i][251, :]
        slice_b = b[i][251, :]
        img = plt.plot(slice_r, 'r')
        img = plt.plot(slice_g, 'g')
        img = plt.plot(slice_b, 'b')
        pos += 1
    plt.show()


if __name__ == '__main__':
    distances = [5,2,8e-2, 5e-2, 3e-2, 2e-2,1e-2]
    resolution = [501,501]
    pixel_pitch = [6.4e-6, 6.4e-6]
    lens_type = ['simple', 'fresnel']
    prop_type = ['fresnel', 'asm']
    folder = 'lens_fresnel_prop_asm'
    # smallest focal distance ppl use = 14 mm
    focal_lengths = [5e-2]
    r_cutoffs = [120]

    lens_idx = 1
    prop_idx = 0

    # for k in range(len(r_cutoffs)):
    starttime = time.time()

    fl = focal_lengths[0]

    print('Generating PSFs for f ', fl)
    g = get_psfs(distances, focal_length=fl, wave_length=wave_lengths[1],
                 physical_size=pixel_pitch[0]*resolution[0],
                 refractive_idc=refractive_idcs[1],
                 lens_type=lens_type[lens_idx],
                 prop_type=prop_type[prop_idx],
                 r_cutoff=r_cutoffs[0])

    # plot_full_psf(psfs = g, num_plots=len(distances),f=fl)
    for i in range(len(distances)):
        plt.figure()
        plt.imshow(g[i])
        plt.colorbar()
        plt.show()


    # center = g[0]
    # middle = resolution[0] // 2 + 1
    # center = center[middle - 1:middle + 2, middle - 1:middle + 2]
    #
    #
    # print(time.time()-starttime)
    # plt.figure()
    # plt.imshow(center)
    # plt.colorbar()
    # plt.show()
    # title  = 'fresnel_1'
    # plt.savefig(title)
    # np.save('fresnel_psf5_1', center)
    #
    # center = g[1]
    # middle = resolution[0] // 2 + 1
    # center = center[middle - 1:middle + 2, middle - 1:middle + 2]
    #
    #
    # print(time.time()-starttime)
    # plt.figure()
    # plt.imshow(center)
    # plt.colorbar()
    # plt.show()
    # title  = 'fresnel_5e-2'
    # plt.savefig(title)
    # np.save('fresnel_psf5_5e-2',center)

    # title  = 'fresnel_' + str(nums[k])
    # plt.savefig(title)
    # center = center.numpy()
    # file = title+'.npy'
    # np.save(file,center)


        # title = directory + folder \
        #             + 'fullPSF_f' + str(fl) \
        #             +'_distance_' + str(distances[i]) \
        #             + '_lens_' + str(lens_type[lens_idx]) \
        #             + '_prop_' + str(prop_type[prop_idx]) + '.png'
        # plt.savefig(title, bbox_inches='tight')
        #
        # plot_side_psf(r,g,b,num_plots)
        #
        # title = directory + folder \
        #         + 'sidePSF_f' + str(fl) \
        #         + 'lens_' + lens_type[lens_idx] \
        #         + '_prop_' + prop_type[prop_idx] + '.png'
        # plt.savefig(title, bbox_inches='tight')