"""
Custom optics functions written PyTorch
Author: Cindy Nguyen
"""

import utils
import torch
import numpy as np
from propagation import Propagation
import torch.nn
import optics


def wiener_filter(img, psf, K):
    """ Performs Wiener filtering on a single channel
    :param img: pytorch tensor of image (N,C,H,W)
    :param psf: pytorch tensor of psf (H,W)
    :param K: damping factor (can be input through hyps or learned)
    :return: Wiener filtered image in one channel (N,C,H,W)
    """
    img = img.cuda()
    psf = psf.cuda()
    imag = torch.zeros(img.shape).cuda()
    img = utils.stack_complex(img,imag)
    img_fft = torch.fft(utils.ifftshift(img),2)
    img_fft = img_fft.cuda()

    otf = psf2otf(psf, output_size=img.shape[2:4])
    otf = torch.stack((otf,otf,otf),0)
    otf = torch.unsqueeze(otf, 0)

    conj_otf = utils.conj(otf)

    otf_img = utils.mul_complex(conj_otf,img_fft)

    denominator = abs_complex(otf)
    denominator[:, :, :, :, 0] += K
    product = utils.div_complex(otf_img, denominator)
    # filtered = utils.ifftshift(torch.ifft(product,2))
    # filtered = torch.clamp(filtered, min=1e-5)

    return otf_img[:,:,:,:,0]
    #return filtered[:,:,:,:,0]


def convolve_img(image, psf):
    """Convolves image with a PSF kernel, convolves on each color channel
    :param image: pytorch tensor of image (B,N,H,W)
    :param psf: pytorch tensor of psf (H,W)
    :return: final convolved image (B,N,H,W)
    """
    image = image.cpu()
    psf = torch.stack((psf, psf, psf), 0)
    psf = torch.unsqueeze(psf, 0)
    psf_stack = utils.stack_complex(psf, torch.zeros(psf.shape))
    img_stack = utils.stack_complex(image, torch.zeros(image.shape))
    convolved = utils.conv_fft(img_stack, psf_stack, padval=0)
    return convolved[:,:,:,:,0]

def circular_aperture(input_field, r_cutoff):
    """
    :param input_field: (H,W,2) - input field
    :param r_cutoff: int or None - radius cutoff for incoming light field
    :return: Light field filtered by the aperture
    """
    input_shape = input_field.shape
    [x, y] = np.mgrid[-(input_shape[0] // 2): (input_shape[0] + 1) // 2,
             -(input_shape[1] // 2):(input_shape[1] + 1) // 2].astype(np.float64)
    if r_cutoff is None:
        r_cutoff = np.amax(x)
    r = np.sqrt(x ** 2 + y ** 2)
    aperture = (r < r_cutoff)
    aperture = torch.Tensor(aperture)
    aperture = utils.stack_complex(aperture, aperture)
    return aperture * input_field


def propagate_through_lens(input_field, phase_delay):
    """
    Provides complex valued wave field upon hitting an optical element
    :param input_field: (H,W) tensor of phase delay of optical element
    :param phase_delay: (H,W) tensor of incoming light field
    :return: (H,W,2) complex valued incident light field
    """
    real, imag = utils.polar_to_rect(1, phase_delay)
    phase_delay = utils.stack_complex(real, imag)

    input_field = utils.stack_complex(input_field,
                                      torch.zeros(input_field.shape))
    return utils.mul_complex(input_field.cpu(), phase_delay.cpu())


def heightmap_to_psf(hyps, height_map):
    resolution = hyps['resolution']
    focal_length = hyps['focal_length']
    wavelength = hyps['wavelength']
    pixel_pitch = hyps['pixel_pitch']
    refractive_idc = hyps['refractive_idc']
    r_cutoff = hyps['r_cutoff']

    input_field = torch.ones((resolution,resolution))

    phase_delay = utils.heightmap_to_phase(height_map,
                                           wavelength,
                                           refractive_idc)

    field = propagate_through_lens(input_field, phase_delay)

    field = circular_aperture(field, r_cutoff)

    # propagate field from aperture to sensor
    element = Propagation(kernel_type='fresnel',
                          propagation_distances=focal_length,
                          slm_resolution=[resolution, resolution],
                          slm_pixel_pitch=[pixel_pitch, pixel_pitch],
                          wavelength=wavelength)
    field = element.forward(field)
    psf = utils.field_to_intensity(field)
    psf /= psf.sum()
    return psf.cuda()


def fspecial_gauss(size, sigma):
    """
    Function to mimic the 'fspecial' gaussian MATLAB function
    :param size: int - size of blur filter
    :param sigma: float - standard deviation of blur
    :return: normalized blur filter
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return torch.Tensor(g/g.sum())


def heightmap_initializer(focal_length,
                          resolution=1248,
                          pixel_pitch=6.4e-6,
                          refractive_idc=1.43,
                          wavelength=530e-9,
                          init_lens='fresnel'):
    """
    Initialize heightmap before training
    :param focal_length: float - distance between phase mask and sensor
    :param resolution: int - size of phase mask
    :param pixel_pitch: float - pixel size of phase mask
    :param refractive_idc: float - refractive index of phase mask
    :param wavelength: float - wavelength of light
    :param init_lens: str - type of lens to initialize
    :return:
    """

    if init_lens == 'fresnel' or init_lens == 'plano':
        convex_radius = (refractive_idc - 1.) * focal_length  # based on lens maker formula

        N = resolution
        M = resolution
        [x, y] = np.mgrid[-(N // 2): (N + 1) // 2,
                 -(M // 2):(M + 1) // 2].astype(np.float64)

        x = x * pixel_pitch
        y = y * pixel_pitch

        # get lens thickness by paraxial approximations
        heightmap = -(x ** 2 + y ** 2) / 2. * (1. / convex_radius)
        if init_lens == 'fresnel':
            phases = utils.heightmap_to_phase(heightmap, wavelength, refractive_idc)
            fresnel = optics.simple_to_fresnel_lens(phases)
            heightmap = utils.phase_to_heightmap(fresnel, wavelength, refractive_idc)

    else:
        heightmap = torch.rand((resolution, resolution)) * pixel_pitch
        gauss_filter = fspecial_gauss(10, 5)

        heightmap = utils.stack_complex(torch.real(heightmap), torch.imag(heightmap))
        gauss_filter = utils.stack_complex(torch.real(gauss_filter), torch.imag(gauss_filter))
        heightmap = utils.conv_fft(heightmap, gauss_filter)
        heightmap = heightmap[:,:,0]

    return torch.Tensor(heightmap)


def psf2otf(input_filter, output_size):
    """
    Converts PSF to OTF that is same size as output_size
    :param input_filter: (H,W) PSF
    :param output_size: [int, int] - size of output filter
    :return: OTF (H,W)
    """
    fh,fw = input_filter.shape

    padder = torch.nn.ZeroPad2d((0, output_size[1]-fw, 0, output_size[0]-fh))
    padded_filter = padder(input_filter)

    # shift left
    left = padded_filter[:,0:(fw-1)//2]
    right = padded_filter[:,(fw-1)//2:]
    padded = torch.cat([right, left], 1)

    # shift down
    up = padded[0:(fh-1)//2,:]
    down = padded[(fh-1)//2:,:]
    padded = torch.cat([down, up], 0)

    tmp = utils.stack_complex(padded.cuda(), torch.zeros(padded.shape).cuda())
    tmp = torch.fft(tmp,2)
    return tmp.cuda()


def abs_complex(input_field):
    """
    Takes absolute value of complex input field
    :param input_field: tensor of size (B,C,H,W,2), last dimension is
    real and imag
    :return: absolute value of complex tensor (B,C,H,W,2)
    """
    real, imag = utils.unstack_complex(input_field)
    real = real ** 2 + imag ** 2
    imag = torch.zeros(real.shape)
    return utils.stack_complex(real.cuda(),imag.cuda())


def simple_to_fresnel_lens(phase_delay):
    """
    Converts a plano convex lens phase delay to a Fresnel phase delay
    through 2*pi phase wrapping
    :param phase_delay: (H,W) phase delay of plano convex lens
    :return: phase delay of a Fresnel lens
    """
    phase_delay -= phase_delay.min()
    return (phase_delay) % (2 * np.pi) - 2 * np.pi
