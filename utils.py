"""A collection of functions for use with complex numbers in pytorch

Author: Nitish Padmanaban
"""

import numpy as np
import torch


def stack_complex(real_mag, imag_phase):
    return torch.stack((real_mag, imag_phase), -1)


def unstack_complex(stacked_array):
    return stacked_array[..., 0], stacked_array[..., 1]


def heightmap_to_phase(height, wavelength, refractive_index):
    return height * (2 * np.pi / wavelength) * (refractive_index - 1)


def phase_to_heightmap(phase, wavelength, refractive_index):
    return phase / (2 * np.pi / wavelength) / (refractive_index - 1)


def rect_to_polar(real, imag):
    mag = torch.pow(real**2 + imag**2, 0.5)
    ang = torch.atan2(imag, real)
    return mag, ang


def polar_to_rect(mag, ang):
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag


def rect_to_polar_stacked(real_imag):
    mag, ang = rect_to_polar(*unstack_complex(real_imag))
    return stack_complex(mag, ang)


def polar_to_rect_stacked(mag_ang):
    real, imag = polar_to_rect(*unstack_complex(mag_ang))
    return stack_complex(real, imag)


def field_to_intensity(real_imag):
    return (real_imag ** 2).sum(-1)


def field_to_intensity_polar(mag_ang):
    return mag_ang[..., 0] ** 2


def conj(real_imag):
    # also works the same for mag_ang representation
    real, imag = unstack_complex(real_imag)
    return stack_complex(real, -imag)


def mul_complex(field1, field2):
    real1, imag1 = unstack_complex(field1)
    real2, imag2 = unstack_complex(field2)

    real = real1 * real2 - imag1 * imag2
    imag = real1 * imag2 + imag1 * real2

    return stack_complex(real, imag)


def mul_complex_polar(field1, field2):
    mag1, ang1 = unstack_complex(field1)
    mag2, ang2 = unstack_complex(field2)

    mag = mag1 * mag2
    ang = ang1 + ang2

    over = ang > np.pi
    ang[over].sub_(2 * np.pi)
    under = ang <= -np.pi
    ang[under].add_(2 * np.pi)

    return stack_complex(mag, ang)


def div_complex(field1, field2):
    real1, imag1 = unstack_complex(field1)
    real2, imag2 = unstack_complex(field2)

    mag_squared = (real2 ** 2) + (imag2 ** 2)

    real = (real1 * real2 + imag1 * imag2) / mag_squared
    imag = (-real1 * imag2 + imag1 * real2) / mag_squared

    return stack_complex(real, imag)


def div_complex_polar(field1, field2):
    mag1, ang1 = unstack_complex(field1)
    mag2, ang2 = unstack_complex(field2)

    mag = mag1 / mag2
    ang = ang1 - ang2

    over = ang > np.pi
    ang[over].sub_(2 * np.pi)
    under = ang <= -np.pi
    ang[under].add_(2 * np.pi)

    return stack_complex(mag, ang)


def recip_complex(field):
    real, imag = unstack_complex(field)

    mag_squared = (real ** 2) + (imag ** 2)

    real_inv = real / mag_squared
    imag_inv = -imag / mag_squared

    return stack_complex(real_inv, imag_inv)


def recip_complex_polar(field):
    mag, ang = unstack_complex(field)
    return stack_complex(1 / mag, -ang)


def conv_fft(img_real_imag, kernel_real_imag, padval=0):
    img_pad, kernel_pad, output_pad = conv_pad_sizes(img_real_imag.shape,
                                                     kernel_real_imag.shape)

    # fft
    img_fft = fft(img_real_imag, pad=img_pad, padval=padval)
    kernel_fft = fft(kernel_real_imag, pad=kernel_pad, padval=0)

    # ifft, using img_pad to bring output to img input size
    return ifft(mul_complex(img_fft, kernel_fft), pad=output_pad)


def conv_fft_polar(img_mag_ang, kernel_mag_ang, padval=0):
    img_pad, kernel_pad, output_pad = conv_pad_sizes(img_mag_ang.shape,
                                                     kernel_mag_ang.shape)

    # fft
    img_fft = fft_polar(img_mag_ang, pad=img_pad, padval=padval)
    kernel_fft = fft_polar(kernel_mag_ang, pad=kernel_pad, padval=0)

    # ifft, using img_pad to bring output to img input size
    return ifft_polar(mul_complex_polar(img_fft, kernel_fft), pad=output_pad)


def fft(real_imag, ndims=2, normalized=False, pad=None, padval=0):
    if pad is not None:
        real_imag = pad_stacked(real_imag, pad, padval=padval)
    return fftshift(torch.fft(ifftshift(real_imag, ndims), ndims,
                              normalized=normalized), ndims)


def fft_polar(mag_ang, ndims=2, normalized=False, pad=None, padval=0):
    real_imag = polar_to_rect_stacked(mag_ang)
    real_imag_fft = fft(real_imag, ndims, normalized, pad, padval)
    return rect_to_polar_stacked(real_imag_fft)


def ifft(real_imag, ndims=2, normalized=False, pad=None):
    transformed = fftshift(torch.ifft(ifftshift(real_imag, ndims), ndims,
                                      normalized=normalized), ndims)
    if pad is not None:
        transformed = crop(transformed, pad)

    return transformed


def ifft_polar(mag_ang, ndims=2, normalized=False, pad=None):
    real_imag = polar_to_rect_stacked(mag_ang)
    real_imag_ifft = ifft(real_imag, ndims, normalized, pad)
    return rect_to_polar_stacked(real_imag_ifft)


def fftshift(array, ndims=2, invert=False):
    shift_adjust = 0 if invert else 1

    # skips the last dimension, assuming stacked fft output
    if ndims >= 1:
        shift_len = (array.shape[-2] + shift_adjust) // 2
        array = torch.cat((array[..., shift_len:, :],
                           array[..., :shift_len, :]), -2)
    if ndims >= 2:
        shift_len = (array.shape[-3] + shift_adjust) // 2
        array = torch.cat((array[..., shift_len:, :, :],
                           array[..., :shift_len, :, :]), -3)
    if ndims == 3:
        shift_len = (array.shape[-4] + shift_adjust) // 2
        array = torch.cat((array[..., shift_len:, :, :, :],
                           array[..., :shift_len, :, :, :]), -4)
    return array


def ifftshift(array, ndims=2):
    return fftshift(array, ndims, invert=True)


def conv_pad_sizes(image_shape, kernel_shape):
    # skips the last dimension, assuming stacked fft output
    # minimum required padding is to img.shape + kernel.shape - 1
    # padding based on matching fftconvolve output

    # when kernels are even, padding the extra 1 before/after matters
    img_pad_end = (1 - ((kernel_shape[-2] % 2) | (image_shape[-2] % 2)),
                   1 - ((kernel_shape[-3] % 2) | (image_shape[-3] % 2)))

    image_pad = ((kernel_shape[-2] - img_pad_end[0]) // 2,
                 (kernel_shape[-2] - 1 + img_pad_end[0]) // 2,
                 (kernel_shape[-3] - img_pad_end[1]) // 2,
                 (kernel_shape[-3] - 1 + img_pad_end[1]) // 2)
    kernel_pad = (image_shape[-2] // 2, (image_shape[-2] - 1) // 2,
                  image_shape[-3] // 2, (image_shape[-3] - 1) // 2)
    output_pad = ((kernel_shape[-2] - 1) // 2, kernel_shape[-2] // 2,
                  (kernel_shape[-3] - 1) // 2, kernel_shape[-3] // 2)
    return image_pad, kernel_pad, output_pad


def pad_stacked(field, pad_width, padval=0):
    if padval == 0:
        pad_width = (0, 0, *pad_width)  # add 0 padding for stacked dimension
        return torch.nn.functional.pad(field, pad_width)
    else:
        if isinstance(padval, torch.Tensor):
            padval = padval.item()

        real, imag = unstack_complex(field)
        real = torch.nn.functional.pad(real, pad_width, value=padval)
        imag = torch.nn.functional.pad(imag, pad_width, value=0)
        return stack_complex(real, imag)


def crop(array, pad):
    # skips the last dimension, assuming stacked fft output
    if len(pad) >= 2 and (pad[0] or pad[1]):
        if pad[1]:
            array = array[..., pad[0]:-pad[1], :]
        else:
            array = array[..., pad[0]:, :]

    if len(pad) >= 4 and (pad[2] or pad[3]):
        if pad[3]:
            array = array[..., pad[2]:-pad[3], :, :]
        else:
            array = array[..., pad[2]:, :, :]

    if len(pad) == 6 and (pad[4] or pad[5]):
        if pad[5]:
            array = array[..., pad[4]:-pad[5], :, :, :]
        else:
            array = array[..., pad[4]:, :, :, :]

    return array


def pad_smaller_dims(field, target_shape, pytorch=True, stacked=True, padval=0):
    if pytorch:
        if stacked:
            size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(target_shape) - np.array(field.shape)
        odd_dim = np.array(field.shape) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [int(p)  # convert from np.int64
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            if stacked:
                return pad_stacked(field, pad_axes, padval=padval)
            else:
                return torch.nn.functional.pad(field, pad_axes, value=padval)
        else:
            return np.pad(field, tuple(zip(pad_front, pad_end)), 'constant',
                          constant_values=padval)
    else:
        return field


def crop_larger_dims(field, target_shape, pytorch=True, stacked=True):
    if pytorch:
        if stacked:
            size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(field.shape) - np.array(target_shape)
        odd_dim = np.array(field.shape) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if pytorch and stacked:
            return field[(..., *crop_slices, slice(None))]
        else:
            return field[(..., *crop_slices)]
    else:
        return field

