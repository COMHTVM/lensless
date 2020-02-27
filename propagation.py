"""Functions for propagation through free space

Propagation class initialization options:
    kernel_type: 'fraunhofer' (alias 'fourier'), 'fresnel', 'fresnel_conv',
                 'asm' (alias 'angular_spectrum'), or 'kirchoff'. The transfer
                 function approaches may be more accurate
        fraunhofer: far-field diffraction, purely a Fourier transform
        fresnel: near-field diffraction with Fresnel approximation, implemented
                 as a multiplication with a transfer function in Fourier domain
        fresnel_conv: same as fresnel, but implemented as a convolution with a
                      spatial kernel, via FFT conv for speed
        asm: near-field diffraction with the Angular Spectrum Method,
             implemented as a transfer function. Note that this may have a 1px
             shift relative to the others due to the source paper padding the
             input by an extra pixel (for linear convolution) for derivations
        kirchoff: near-field diffractoin with the Kirchoff equations,
                  implemented with a spatial kernel
    propagation_distances: distance  or distances from SLM to image plane.
                           Accepts scalars or lists.
    slm_resolution: number of pixels on SLM
    slm_pixel_pitch: size of pixels on SLM.
    image_resolution: number of sampling locations at image plane (optional,
                      default matches SLM resolution)
    wavelength: laser wavelength, (optional, default 532e-9).
    propagation_parameters: override parameters for kernel/transfer function
                            construction. Optional. Possible parameters, with
                            defaults given:
        # for all methods
        'padding_type', 'zero':  pad complex field with 'median' or 'zero'.
                                 Using median may have less ringing, but zero
                                 is probably more accurate
        # for the spatial kernel convolution methods
        'circular_prop_mask', True:  circular mask for propagation kernels, for
                                     bandlimiting the phase function
        'apodize_kernel', True:  smooth the circular mask
        'apodization_width', 50:  width of cosine dropoff at edge, in pixels
        'prop_mask_fraction', 1:  artificially reduces the size of propagation
                                  mask (e.g., 2 will use half the radius)
        'normalize_output', True:  forces output field to have the same average
                                   amplitudes as the input when True. Only valid
                                   when using a single propagation distance
        # for the transfer function multiplication methods
        'circular_padding', False: doesn't pad the field when True, resulting in
                                   implicit circular padding in the Fourier
                                   domain for the input field. May reduce
                                   ringing at the edges
        'normalize_output', False: same as for the spatial kernel methods, but
                                   defaults to False because the transfer
                                   functions do a better job at energy
                                   preservation by default
        # only for the Angular Spectrum Method
        'extra_pixel', True: when not using circular_padding, i.e., for a linear
                             convolution, pad one extra pixel more than required
                             (i.e., linear conv to length a + b instead of the
                             minimum valid a + b - 1). The derivation from
                             Matsushima and Shimobaba (2009) has an extra pixel,
                             may not be correct without it, but set if the pixel
                             shift is important
        # only for Fraunhofer
        'fraunhofer_crop_image', True:  when resolution changes, crop image
                                        plane instead of SLM plane, details in
                                        __init__ for FraunhoferPropagation
        # only for Fraunhofer with multiple distances
        'focal_length', no default:  required to determine plane for Fourier
                                     relationship (e.g., lens focal length)
                                     relative to which the other distances are
                                     propagated.
    device: torch parameter for the device to place the convolution kernel on.
            If not given, will default to the device of the input_field.

Propagation.forward and Propagation.backward:
    input_field: complex field at starting plane (e.g. SLM for foward)

    Returns: output_field at the ending plane matching the specified resolution
             (for single distance) or output_fields, a dictionary of fields at
             each propagation distance (keys are distances)

All units are in meters and radians unless explicitly stated as otherwise.
Terms for resolution are in ij (matrix) order, not xy (cartesian) order.

input_field should be a torch Tensor, everything else can be either numpy or
native python types. input_field is assumed to be a stack of [real, imag] for
input to the fft (see the torch.fft implementation for details). The
output_field follows the same convention.

Example: Propagate some input_field by 10cm with Fresnel approx, 5um pixel pitch
        on the SLM, with a 1080p SLM and image size equal to it
    prop = Propagation('fresnel', 10e-2, [1080, 1920], [5e-6, 5e-6])
    output_field = prop.forward(input_field)
    output_field = prop.backward(input_field)

Example: Propagate some input_field by to multiple distances, using Kirchhoff
        propagation.
    prop = Propagation('kirchhoff', [10e-2, 20e-2, 30e-2], [1080, 1920],
                       [5e-6, 5e-6])

Example: Setting non-default parameters, e.g. wavelength of 632nm, image
        resolution of 720p, image sampling of 8um, some of the extra propagation
        parameters, or device to gpu 0
    propagation_parameters = {'circular_prop_mask': True,
                              'apodize_kernel': True}
    prop = Propagation('fresnel', 10e-2, [1080, 1920], [5e-6, 5e-6],
                       [720, 1280], [8e-6, 8e-6], 632e-9,
                       propagation_parameters, torch.device('cuda:0'))
    # or with named parameters
    prop = Propagation(kernel_type='fresnel',
                       propagation_distances=10e-2,
                       slm_resolution=[1080, 1920],
                       slm_pixel_pitch=[5e-6, 5e-6],
                       image_resolution=[720, 1280],
                       wavelength=632e-9,
                       propagation_parameters=propagation_parameters,
                       device=torch.device('cuda:0'))

Example: Other propagation kernels, alternate ways to define it
    prop = Propagation('Fresnel', ...)  # not case sensitive
    prop = Propagation('fraunhofer', ...)  # Fraunhofer
    prop = Propagation('asm', ...)  # Angular Spectrum Method

Author: Nitish Padmanaban
"""

import numpy as np
import torch
import torch.nn as nn
import warnings
import utils


class Propagation:
    """Convenience class for using different propagation kernels and sets of
    propagation distances"""
    def __new__(cls, kernel_type, propagation_distances, slm_resolution,
                slm_pixel_pitch, image_resolution=None, wavelength=532e-9,
                propagation_parameters=None, device=None):
        # process input types for propagation distances
        if isinstance(propagation_distances, (np.ndarray, torch.Tensor)):
            propagation_distances = propagation_distances.flatten().tolist()
        # singleton lists should be made into scalars
        if (isinstance(propagation_distances, (tuple, list))
                and len(propagation_distances) == 1):
            propagation_distances = propagation_distances[0]

        # scalar means this is a single distance propagation
        if not isinstance(propagation_distances, (tuple, list)):
            cls_out = {'fresnel': FresnelPropagation,
                       'fresnel_conv': FresnelConvPropagation,
                       'asm': AngularSpectrumPropagation,
                       'angular_spectrum': AngularSpectrumPropagation,
                       'kirchhoff': KirchhoffPropagation,
                       'fraunhofer': FraunhoferPropagation,
                       'fourier': FraunhoferPropagation}[kernel_type.lower()]
            return cls_out(propagation_distances, slm_resolution,
                           slm_pixel_pitch, image_resolution, wavelength,
                           propagation_parameters, device)
        else:
            return MultiDistancePropagation(
                kernel_type, propagation_distances, slm_resolution,
                slm_pixel_pitch, image_resolution, wavelength,
                propagation_parameters, device)


class PropagationBase(nn.Module):
    image_native_pitch = None

    """Interface for propagation functions, with some shared functions"""
    def __init__(self, propagation_distance, slm_resolution, slm_pixel_pitch,
                 image_resolution=None, wavelength=532e-9,
                 propagation_parameters=None, device=None):
        super().__init__()
        self.slm_resolution = np.array(slm_resolution)
        self.slm_pixel_pitch = np.array(slm_pixel_pitch)
        self.propagation_distance = propagation_distance
        self.wavelength = wavelength
        self.dev = device

        # default image dimensions to slm dimensions
        if image_resolution is None:
            self.image_resolution = self.slm_resolution
        else:
            self.image_resolution = np.array(image_resolution)

        # native image sampling matches slm pitch, unless overridden by a
        # deriving class (e.g. FraunhoferPropagation)
        if self.image_native_pitch is None:
            self.image_native_pitch = self.slm_pixel_pitch

        # set image pixel pitch to native image sampling
        self.image_pixel_pitch = self.image_native_pitch

        # physical size of planes in meters
        self.slm_size = self.slm_pixel_pitch * self.slm_resolution
        self.image_size = self.image_pixel_pitch * self.image_resolution

        # dictionary for extra parameters particular to base class
        self.propagation_parameters = propagation_parameters
        if self.propagation_parameters is None:
            self.propagation_parameters = {}

        # set default for padding type when convolving
        try:
            self.padding_type = self.propagation_parameters.pop('padding_type')
        except KeyError:
            self.padding_type = 'zero'

    def forward(self, input_field):
        """Returns output_field, which is input_field propagated by
        propagation_distance, from slm_resolution to image_resolution"""
        raise NotImplementedError('Must implement in derived class')

    def backward(self, input_field):
        """Returns output_field, which is input_field propagated by
        -propagation_distance, from image_resolution to slm_resolution"""
        raise NotImplementedError('Must implement in derived class')

    def to(self, *args, **kwargs):
        """Moves non-parameter tensors needed for propagation to device

        Also updates the internal self.dev added to this class
        """
        slf = super().to(*args, **kwargs)

        device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device_arg is not None:
            slf.dev = device_arg

        return slf

    def pad_smaller_dims(self, field, target_shape, pytorch=True, padval=None):
        if padval is None:
            padval = self.get_pad_value(field, pytorch)
        return utils.pad_smaller_dims(field, target_shape, pytorch,
                                      padval=padval)

    def crop_larger_dims(self, field, target_shape, pytorch=True):
        return utils.crop_larger_dims(field, target_shape, pytorch)

    def get_pad_value(self, field, pytorch=True):
        if self.padding_type == 'zero':
            return 0
        elif self.padding_type == 'median':
            if pytorch:
                return torch.median(stacked_abs(field))
            else:
                return np.median(np.abs(field))
        else:
            raise ValueError('Unknown padding type')


class NearFieldConvPropagationBase(PropagationBase):
    """Defines functions shared across propagation near field approximations
    based on convolving a kernel
    """
    def __init__(self, propagation_distance, slm_resolution, slm_pixel_pitch,
                 image_resolution=None, wavelength=532e-9,
                 propagation_parameters=None, device=None):
        super().__init__(propagation_distance, slm_resolution, slm_pixel_pitch,
                         image_resolution, wavelength, propagation_parameters,
                         device)
        # diffraction pattern calculations
        self.max_diffraction_angle = np.arcsin(wavelength
                                               / self.slm_pixel_pitch / 2)
        self.prop_mask_radius = (propagation_distance
                                 * np.tan(self.max_diffraction_angle))

        # limit zone plate to maximum usable size
        slm_diagonal = np.sqrt((self.slm_size**2).sum())
        image_diagonal = np.sqrt((self.image_size**2).sum())
        max_usable_distance = slm_diagonal / 2 + image_diagonal / 2
        self.prop_mask_radius = np.minimum(self.prop_mask_radius,
                                           max_usable_distance)

        # force input and output of forward/backward
        # operations to have the same absolute sum
        try:
            self.normalize_output = self.propagation_parameters.pop(
                'normalize_output')
        except KeyError:
            self.normalize_output = True

        # sets self.foward_kernel and self.backward_kernel
        self.compute_conv_kernels(**self.propagation_parameters)

        if self.dev is not None:
            self.forward_kernel = self.forward_kernel.to(self.dev)
            self.backward_kernel = self.backward_kernel.to(self.dev)

    def compute_conv_kernels(self, *, circular_prop_mask=True,
                             apodize_kernel=True, apodization_width=50,
                             prop_mask_fraction=1., **kwargs):
        # sampling positions along the x and y dims
        coords_x = np.arange(self.slm_pixel_pitch[1],
                             self.prop_mask_radius[1] / prop_mask_fraction,
                             self.slm_pixel_pitch[1])
        coords_x = np.concatenate((-coords_x[::-1], [0], coords_x))
        coords_y = np.arange(self.slm_pixel_pitch[0],
                             self.prop_mask_radius[0] / prop_mask_fraction,
                             self.slm_pixel_pitch[0])
        coords_y = np.concatenate((-coords_y[::-1], [0], coords_y))

        samples_x, samples_y = np.meshgrid(coords_x, coords_y)

        # compute complex forward propagation at sampled points
        forward = self.forward_prop_at_points(samples_x, samples_y)

        if circular_prop_mask:
            forward = self.apply_circular_mask(forward,
                                               np.sqrt(samples_x**2
                                                       + samples_y**2),
                                               apodize_kernel,
                                               apodization_width)

        # rescale for approx energy preservation even when normalization off
        # forward *= self.wavelength / np.sum(self.slm_resolution)
        forward /= np.sum(np.abs(forward))

        # convert to stacked real and imaginary for pytorch fft format
        forward_stacked = np.stack((np.real(forward), np.imag(forward)), -1)
        self.forward_kernel = torch.from_numpy(forward_stacked).float()
        # reverse prop is just the conjugate
        backward_stacked = np.stack((np.real(forward), -np.imag(forward)), -1)
        self.backward_kernel = torch.from_numpy(backward_stacked).float()

    def forward_prop_at_points(self, samples_x, samples_y):
        """computes the convolution kernel for the deriving class's
        particular approximation
        """
        raise NotImplementedError('Must implement in derived class')

    def apply_circular_mask(self, pattern, distances, apodize=True,
                            apodization_width=50):
        # furthest point along smaller dimension, max usable radius
        max_radius = min(distances[0, :].min(), distances[:, 0].min())

        if apodize:
            # set the width of apodization based on the wider pixel pitch
            pixel_pitch = max(self.slm_pixel_pitch)
            apodization_width *= pixel_pitch

            if apodization_width > max_radius:
                apodization_width = max_radius

            # ramp that rises to 1 over a length of apodization_width
            normalized_edge_dist = (max_radius - distances) / apodization_width
            normalized_edge_dist = normalized_edge_dist.clip(min=0, max=1)

            # convert ramp to smooth cos
            mask = 1 / 2 + np.cos(np.pi * normalized_edge_dist - np.pi) / 2
            mask /= mask.max()  # make sure it's max 1, probably not needed
        else:
            mask = (distances <= max_radius).astype(np.float64)

        return pattern * mask

    def forward(self, input_field):
        # force kernel device to input's device if this module specifies nothing
        if (self.dev is None
                and self.forward_kernel.device != input_field.device):
            self.forward_kernel = self.forward_kernel.to(input_field.device)

        if self.normalize_output:
            input_magnitude_sum = magnitude_sum(input_field)

        padval = self.get_pad_value(input_field)
        input_padded = self.pad_smaller_dims(input_field, self.image_resolution,
                                             padval=padval)
        output_field = utils.conv_fft(input_padded, self.forward_kernel,
                                      padval=padval)
        output_cropped = self.crop_larger_dims(output_field,
                                               self.image_resolution)
        if self.normalize_output:
            output_magnitude_sum = magnitude_sum(output_cropped)
            output_cropped = output_cropped * (input_magnitude_sum
                                               / output_magnitude_sum)

        return output_cropped

    def backward(self, input_field):
        # force kernel device to input's device if this module specifies nothing
        if (self.dev is None
                and self.backward_kernel.device != input_field.device):
            self.backward_kernel = self.backward_kernel.to(input_field.device)

        if self.normalize_output:
            input_magnitude_sum = magnitude_sum(input_field)

        padval = self.get_pad_value(input_field)
        input_padded = self.pad_smaller_dims(input_field, self.slm_resolution,
                                             padval=padval)
        output_field = utils.conv_fft(input_padded, self.backward_kernel,
                                      padval=padval)
        output_cropped = self.crop_larger_dims(output_field,
                                               self.slm_resolution)
        if self.normalize_output:
            output_magnitude_sum = magnitude_sum(output_cropped)
            output_cropped = output_cropped * (input_magnitude_sum
                                               / output_magnitude_sum)

        return output_cropped

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)

        if slf.dev is not None:
            slf.forward_kernel = slf.forward_kernel.to(slf.dev)
            slf.backward_kernel = slf.backward_kernel.to(slf.dev)

        return slf


class NearFieldTransferFnPropagationBase(PropagationBase):
    """Defines functions shared across propagation near field approximations
    based on applying the transfer function in Fourier domain
    """
    def __init__(self, propagation_distance, slm_resolution, slm_pixel_pitch,
                 image_resolution=None, wavelength=532e-9,
                 propagation_parameters=None, device=None):
        super().__init__(propagation_distance, slm_resolution, slm_pixel_pitch,
                         image_resolution, wavelength, propagation_parameters,
                         device)
        # force input and output of forward/backward
        # operations to have the same absolute sum
        try:
            self.normalize_output = self.propagation_parameters.pop(
                'normalize_output')
        except KeyError:
            self.normalize_output = False

        # sets self.foward_kernel and self.backward_kernel
        self.compute_transfer_fn(**self.propagation_parameters)

        if self.dev is not None:
            self.forward_transfer_fn = self.forward_transfer_fn.to(self.dev)
            self.backward_transfer_fn = self.backward_transfer_fn.to(self.dev)

    def compute_transfer_fn(self, *, circular_padding=False, **kwargs):
        """computes the Fourier transfer function for the deriving class's
        particular approximation
        """
        raise NotImplementedError('Must implement in derived class')

    def forward(self, input_field):
        # force transfer function device to input's device if this module
        # specifies nothing
        if (self.dev is None
                and self.forward_transfer_fn.device != input_field.device):
            self.forward_transfer_fn = self.forward_transfer_fn.to(
                input_field.device)

        if self.normalize_output:
            input_magnitude_sum = magnitude_sum(input_field)

        # compute Fourier transform of input field
        fourier_input = self.padded_fft(input_field)

        # apply transfer function for forward prop
        fourier_output = utils.mul_complex(fourier_input,
                                           self.forward_transfer_fn)

        # Fourier transform back to get output
        output_cropped = self.cropped_ifft(fourier_output,
                                           self.image_resolution)

        if self.normalize_output:
            output_magnitude_sum = magnitude_sum(output_cropped)
            output_cropped = output_cropped * (input_magnitude_sum
                                               / output_magnitude_sum)

        return output_cropped

    def backward(self, input_field):
        # force transfer function device to input's device if this module
        # specifies nothing
        if (self.dev is None
                and self.backward_transfer_fn.device != input_field.device):
            self.backward_transfer_fn = self.backward_transfer_fn.to(
                input_field.device)

        if self.normalize_output:
            input_magnitude_sum = magnitude_sum(input_field)

        # compute Fourier transform of input field
        fourier_input = self.padded_fft(input_field)

        # apply transfer function for backward prop
        fourier_output = utils.mul_complex(fourier_input,
                                           self.backward_transfer_fn)

        # Fourier transform back to get output
        output_cropped = self.cropped_ifft(fourier_output, self.slm_resolution)

        if self.normalize_output:
            output_magnitude_sum = magnitude_sum(output_cropped)
            output_cropped = output_cropped * (input_magnitude_sum
                                               / output_magnitude_sum)

        return output_cropped

    def padded_fft(self, input_field):
        input_padded = self.pad_smaller_dims(input_field, self.conv_resolution)
        return utils.fft(input_padded)

    def cropped_ifft(self, fourier_output, output_res):
        output_field = utils.ifft(fourier_output)
        return self.crop_larger_dims(output_field, output_res)

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)

        if slf.dev is not None:
            slf.forward_transfer_fn = slf.forward_transfer_fn.to(slf.dev)
            slf.backward_transfer_fn = slf.backward_transfer_fn.to(slf.dev)

        return slf


class FresnelConvPropagation(NearFieldConvPropagationBase):
    """Implements the Fresnel approximation for the kernel"""
    def forward_prop_at_points(self, samples_x, samples_y):
        # prevent 0
        if abs(self.propagation_distance) < 1e-10:
            prop_dist = -1e-10 if self.propagation_distance < 0 else 1e-10
        else:
            prop_dist = self.propagation_distance
        wave_number = 2 * np.pi / self.wavelength

        # exclude propagation_distance for zero phase at center
        phase_term = ((samples_x**2 + samples_y**2) / (2 * prop_dist))
        # ignore 1/j term
        amplitude_term = 1 / prop_dist / self.wavelength
        return amplitude_term * np.exp(1j * wave_number * phase_term)


class KirchhoffPropagation(NearFieldConvPropagationBase):
    """Implements the Kirchhoff approximation for the kernel"""
    def forward_prop_at_points(self, samples_x, samples_y):
        # prevent 0
        if abs(self.propagation_distance) < 1e-10:
            prop_dist = -1e-10 if self.propagation_distance < 0 else 1e-10
        else:
            prop_dist = self.propagation_distance
        wave_number = 2 * np.pi / self.wavelength

        radius = np.sqrt(prop_dist**2 + samples_x**2 + samples_y**2)
        phase_term = radius - prop_dist  # zero phase at center
        # ignore 1/j term
        amplitude_term = prop_dist / self.wavelength / radius**2
        return amplitude_term * np.exp(1j * wave_number * phase_term)


class FresnelPropagation(NearFieldTransferFnPropagationBase):
    """Implements the Fresnel approximation for the transfer function"""
    def compute_transfer_fn(self, *, circular_padding=False, **kwargs):
        # we always convolve at the size of the larger dimensions
        self.conv_resolution = np.maximum(self.slm_resolution,
                                          self.image_resolution)
        # for linear convolution, otherwise the input
        # field is implicitly circularly padded
        if not circular_padding:
            self.conv_resolution = self.conv_resolution * 2 - 1
        # physical dimensions
        self.conv_size = self.slm_pixel_pitch * self.conv_resolution

        # sampling positions along the x and y dims
        min_coords = -1 / (2 * self.slm_pixel_pitch)
        max_coords = 1 / (2 * self.slm_pixel_pitch) - 1 / self.conv_size

        coords_fx = np.linspace(min_coords[1],
                                max_coords[1],
                                self.conv_resolution[1])
        coords_fy = np.linspace(min_coords[0],
                                max_coords[0],
                                self.conv_resolution[0])

        samples_fx, samples_fy = np.meshgrid(coords_fx, coords_fy)

        forward_phases = (np.pi * -self.propagation_distance * self.wavelength
                          * (samples_fx**2 + samples_fy**2))

        forward = np.exp(1j * forward_phases)

        # convert to stacked real and imaginary for pytorch fft format
        forward_stacked = np.stack((np.real(forward), np.imag(forward)), -1)
        self.forward_transfer_fn = torch.from_numpy(forward_stacked).float()
        # reverse prop is just the conjugate
        backward_stacked = np.stack((np.real(forward), -np.imag(forward)), -1)
        self.backward_transfer_fn = torch.from_numpy(backward_stacked).float()


class AngularSpectrumPropagation(NearFieldTransferFnPropagationBase):
    """Implements the Fresnel approximation for the transfer function"""
    def compute_transfer_fn(self, *, circular_padding=False, extra_pixel=True,
                            **kwargs):
        # we always convolve at the size of the larger dimensions
        self.conv_resolution = np.maximum(self.slm_resolution,
                                          self.image_resolution)
        # for linear convolution, otherwise the input
        # field is implicitly circularly padded
        if not circular_padding:
            self.conv_resolution *= 2
            # Note: Matsushima and Shimobaba (2009) only discuss 2x padding,
            # unclear if this is correct without the extra pixel
            if not extra_pixel:
                self.conv_resolution -= 1
        # physical dimensions
        self.conv_size = self.slm_pixel_pitch * self.conv_resolution

        # sampling positions along the x and y dims
        max_coords = 1 / (2 * self.slm_pixel_pitch) - 0.5 / (2 * self.conv_size)
        coords_fx = np.linspace(-max_coords[1],
                                max_coords[1],
                                self.conv_resolution[1])
        coords_fy = np.linspace(-max_coords[0],
                                max_coords[0],
                                self.conv_resolution[0])

        samples_fx, samples_fy = np.meshgrid(coords_fx, coords_fy)

        forward_phases = (2 * np.pi * self.propagation_distance
                          * np.sqrt(self.wavelength**-2 - (samples_fx**2
                                                           + samples_fy**2)))

        # bandlimit the transfer function, Matsushima and Shimobaba (2009)
        f_max = 1 / np.sqrt((2 * self.propagation_distance / self.conv_size)**2
                            + 1) / self.wavelength
        freq_support = ((np.abs(samples_fx) < f_max[1])
                        & (np.abs(samples_fy) < f_max[0]))

        forward = freq_support * np.exp(1j * forward_phases)

        # convert to stacked real and imaginary for pytorch fft format
        forward_stacked = np.stack((np.real(forward), np.imag(forward)), -1)
        self.forward_transfer_fn = torch.from_numpy(forward_stacked).float()
        # reverse prop is just the conjugate
        backward_stacked = np.stack((np.real(forward), -np.imag(forward)), -1)
        self.backward_transfer_fn = torch.from_numpy(backward_stacked).float()


class FraunhoferPropagation(PropagationBase):
    """Implements Fraunhofer propagation, where lens focal length is given by
    propagation_distance"""
    def __init__(self, propagation_distance, slm_resolution, slm_pixel_pitch,
                 image_resolution=None, wavelength=532e-9,
                 propagation_parameters=None, device=None):
        # Fraunhofer propagation has a different native resolution at image
        # plane, defined by the transform relating the SLM and image planes. It
        # uses frequencies of x/(lambda*f), which changes the sampling density
        self.focal_length = propagation_distance
        # extent of slm
        slm_bandwidth = np.array(slm_pixel_pitch) * np.array(slm_resolution)
        slm_fourier_sampling = 1 / slm_bandwidth
        self.image_native_pitch = (slm_fourier_sampling * wavelength
                                   * self.focal_length)

        super().__init__(propagation_distance, slm_resolution, slm_pixel_pitch,
                         image_resolution, wavelength, propagation_parameters,
                         device)

        # for Fraunhofer propagation, etendue fixes the output physical
        # dimensions based on SLM pixel pitch. For a bigger image resolution, we
        # just pad the SLM field before propagation. For a smaller image, we can
        # either crop first to use part of the SLM to produce a low resolution,
        # but full physical size output, or crop after to use less of the
        # physical area, but keep the high resolution by using the full SLM.
        # Default is to crop the image so that we have more degrees of freedom
        # on the SLM
        try:
            self.fraunhofer_crop_image = self.propagation_parameters.pop(
                'fraunhofer_crop_image')
        except KeyError:
            self.fraunhofer_crop_image = True

    def forward(self, input_field):
        input_padded = self.pad_smaller_dims(input_field, self.image_resolution)

        if self.fraunhofer_crop_image:
            output_field = utils.fft(input_padded, normalized=True)
            return self.crop_larger_dims(output_field, self.image_resolution)
        else:
            input_padded_cropped = self.crop_larger_dims(input_padded,
                                                         self.image_resolution)
            return utils.fft(input_padded_cropped, normalized=True)

    def backward(self, input_field):
        # reverse the operations of the forward field
        if self.fraunhofer_crop_image:
            input_padded = self.pad_smaller_dims(input_field,
                                                 self.slm_resolution)
            output_field = utils.ifft(input_padded, normalized=True)
        else:
            output_field_unpadded = utils.ifft(input_field, normalized=True)
            output_field = self.pad_smaller_dims(output_field_unpadded,
                                                 self.slm_resolution)

        return self.crop_larger_dims(output_field, self.slm_resolution)


class MultiDistancePropagation(nn.Module):
    """Container class that handles propagating to multiple distances"""
    def __init__(self, kernel_type, propagation_distances, slm_resolution,
                 slm_pixel_pitch, image_resolution=None, wavelength=532e-9,
                 propagation_parameters=None, device=None):
        super().__init__()
        self.kernel_type = kernel_type.lower()
        self.slm_resolution = slm_resolution
        self.slm_pixel_pitch = slm_pixel_pitch
        self.image_resolution = image_resolution
        self.wavelength = wavelength
        self.propagation_parameters = propagation_parameters
        self.dev = device
        if self.propagation_parameters is None:
            self.propagation_parameters = {}

        # for near field distances, turn off internal normalization
        # so it can be applied uniformly accross all distances
        self.propagation_parameters['normalize_output'] = False

        self.has_fraunhofer = kernel_type in ('fraunhofer', 'fourier')

        # process input types for propagation distances
        if isinstance(propagation_distances, (np.ndarray, torch.Tensor)):
            propagation_distances = propagation_distances.flatten().tolist()
        # unique values only
        self.propagation_distances = set(propagation_distances)
        # mappings if modified for Fourier plane
        self.get_original_dist = {d: d for d in self.propagation_distances}
        self.get_internal_dist = {d: d for d in self.propagation_distances}

        # dictionary for the set of propagators
        self.propagators = {}

        if self.has_fraunhofer:
            self.create_fraunhofer_propagator()
            # all other planes will be propagated from the Fourier plane,
            # keeping its resolution and pixel pitch
            self.kernel_type = 'kirchhoff'
            self.create_near_field_propagators(self.fourier_resolution,
                                               self.fourier_pixel_pitch,
                                               None)
        else:
            self.create_near_field_propagators(self.slm_resolution,
                                               self.slm_pixel_pitch,
                                               self.image_resolution)

    def create_near_field_propagators(self, start_resolution, start_pixel_pitch,
                                      image_resolution):
        prop_cls = {'fresnel': FresnelPropagation,
                    'fresnel_conv': FresnelConvPropagation,
                    'asm': AngularSpectrumPropagation,
                    'angular_spectrum': AngularSpectrumPropagation,
                    'kirchhoff': KirchhoffPropagation}[self.kernel_type]
        for d in self.propagation_distances:
            if d == 0:
                continue
            self.propagators[d] = prop_cls(
                d, start_resolution, start_pixel_pitch, image_resolution,
                self.wavelength, self.propagation_parameters.copy(),
                self.dev)

    def create_fraunhofer_propagator(self):
        try:
            self.focal_length = self.propagation_parameters.pop('focal_length')
        except KeyError:
            raise ValueError("Multi-distance Fraunhofer propagation requires "
                             "'focal_length' in propagation_parameters to "
                             "specify which propagation_distance has the "
                             "Fourier relationship.")

        if self.focal_length not in self.propagation_distances:
            warnings.warn('focal_length is not in the list of '
                          'propagation_distances. Add it if you also want '
                          'the Fourier plane output field.')

        # set the propagation distances relative to Fourier plane
        self.get_original_dist = {d - self.focal_length: d
                                  for d in self.propagation_distances}
        self.get_internal_dist = {d: d - self.focal_length
                                  for d in self.propagation_distances}
        # make sure 0 doesn't have a rounding error
        if 0 not in self.get_original_dist:
            zero_value = None
            for d in self.propagation_distances:
                if abs(d) < 1e-10:
                    zero_value = d
                    break
            if zero_value is not None:
                orig_dist = self.get_original_dist.pop(zero_value)
                self.get_original_dist[0] = orig_dist
                self.get_internal_dist[orig_dist] = 0
        # update the propagation distances for internal use
        self.propagation_distances = set(self.get_original_dist.keys())

        # make Fraunhofer propagator
        self.fraunhofer_propagator = FraunhoferPropagation(
            self.focal_length, self.slm_resolution, self.slm_pixel_pitch,
            self.image_resolution, self.wavelength,
            self.propagation_parameters.copy(), self.dev)
        self.fourier_resolution = self.fraunhofer_propagator.image_resolution
        self.fourier_pixel_pitch = (self.fraunhofer_propagator
                                    .image_pixel_pitch)

    def forward(self, input_field):
        # for normalization
        input_magnitude_sum = magnitude_sum(input_field)

        # do Fraunhofer propagation first if needed
        if self.has_fraunhofer:
            input_field = self.fraunhofer_propagator.forward(input_field)

        output_fields = {}
        output_sums = {}
        for d in self.propagation_distances:
            if d == 0:
                output_fields[d] = input_field
            else:
                output_fields[d] = self.propagators[d].forward(input_field)
            output_sums[d] = magnitude_sum(output_fields[d])

        # give the 0 distance layer twice the weight of the highest other layer.
        # This is mainly for the Fraunhofer propagation case, since we want the
        # layers to have the correct relative radiometric fall-off, but the
        # Fourier plane itself would dominate the backprop, so we compensate
        if 0 in self.propagation_distances:
            sum_max = max(output_sums[d] for d in output_sums if d != 0)
            output_fields[0] = output_fields[0] * (2 * sum_max / output_sums[0])
            output_sums[0] = 2 * sum_max

        # normalize output based on input
        output_magnitude_sum = sum(output_sums[d] for d in output_sums)
        scale_factor = (input_magnitude_sum / output_magnitude_sum
                        * len(self.propagation_distances))
        for d in output_fields:
            output_fields[d].mul_(scale_factor)

        # return using original distances as keys
        return {self.get_original_dist[d]: output_fields[d]
                for d in output_fields}

    def backward(self, input_fields):
        input_magnitude_sum = sum(magnitude_sum(input_fields[d])
                                  for d in input_fields)

        output_fields = {}
        output_sums = {}
        for d_orig in input_fields:
            d = self.get_internal_dist[d_orig]
            if d == 0:
                output_fields[d] = input_fields[d_orig]
            else:
                output_fields[d] = self.propagators[d].backward(
                    input_fields[d_orig])
            output_sums[d] = magnitude_sum(output_fields[d])

        # compensate for a 0 distance propagation layer (see self.forward())
        if 0 in self.propagation_distances:
            sum_max = max(output_sums[d] for d in output_sums if d != 0)
            output_fields[0] = output_fields[0] * (2 * sum_max / output_sums[0])
            output_sums[0] = 2 * sum_max

        # combine the fields
        output_field = torch.stack(list(output_fields.values()), -1).sum(-1)

        # reverse Fraunhofer propagation if needed
        if self.has_fraunhofer:
            output_field = self.fraunhofer_propagator.backward(output_field)

        # normalize output based on input
        output_magnitude_sum = magnitude_sum(output_field)
        output_field.mul_(input_magnitude_sum / output_magnitude_sum
                          / len(self.propagation_distances))

        return output_field

    def to(self, *args, **kwargs):
        """Moves non-parameter tensors needed for propagation to device

        Also updates the internal self.dev added to this class
        """
        slf = super().to(*args, **kwargs)

        device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device_arg is not None:
            slf.dev = device_arg

            if slf.has_fraunhofer:
                slf.fraunhofer_propagator.to(slf.dev)

            for d in slf.propagation_distances:
                slf.propagators[d].to(slf.dev)

        return slf


def stacked_abs(field):
    # for a complex field stacked according to pytorch fft format, computes
    # the magnitude for each pixel
    return torch.pow(utils.field_to_intensity(field), 0.5)


def magnitude_sum(field):
    # for a complex field stacked according to pytorch fft format, computes
    # a normalization factor over the magnitudes
    return stacked_abs(field).mean()
