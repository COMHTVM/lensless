import torch
from torch.utils import data
import torchvision
from torchvision.transforms import *
import matplotlib.pyplot as plt
import sys
import torch.nn as nn
import numpy as np
from numpy.fft import fft2, ifft2

sys.path.append("../")
from utils import *

def srgb_to_linear(img):
    return np.where(img <= 0.0031308, 12.92 * img, 1.055 * img ** (0.41666) - 0.055)

def linear_to_srgb(img):
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def psf2otf(input_filter, output_size):
    ''' Convert pytorch tensor PSF into its FFT
    :param input_filter: tensor (height, width)
    :param output_size: (height, width, num_channels) of image
    :return: otf of output_size
    '''
    fh,fw = input_filter.shape

    padder = nn.ZeroPad2d((0, output_size[1]-fw, 0, output_size[0]-fh))
    padded_filter = padder(input_filter)

    # shift left
    left = padded_filter[:,0:(fw-1)//2]
    right = padded_filter[:,(fw-1)//2:]
    padded = torch.cat([right, left], 1)

    # shift down
    up = padded[0:(fh-1)//2,:]
    down = padded[(fh-1)//2:,:]
    padded = torch.cat([down, up], 0)

    # take FFT
    tmp = stack_complex(torch.real(padded), torch.imag(padded))
    tmp = torch.fft(tmp,2)

    return tmp

def convolve_img(image, psf):
    ''' Convolves image with a PSF kernel, convolves on each color channel
    :param image: pytorch tensor of image (num_channels, height, width)
    :param psf: pytorch tensor of psf (height, width)
    :return: final convolved image (num_channels, height, width)
    '''
    final = torch.zeros(image.shape)
    psf = stack_complex(torch.real(psf), torch.imag(psf))
    for i in range(0,3): # iterate over RGB color channels
        channel = torch.Tensor(image[i,:,:])
        channel = torch.stack((torch.real(channel), torch.imag(channel)), -1)
        convolved_image = conv_fft(channel, psf, padval=0)
        convolved_image = field_to_intensity(convolved_image)
        final[i,:,:] = convolved_image
    return final


def inverse_filter(image, psf, K):
    ''' Modified inverse filter with damping factor
    Performs inverse filtering on each channel
    :param image: pytorch tensor (num_channels, height, width)
    :param psf: pytorch tensor of psf (height, width)
    :param K: float damping factor
    :return:
    '''

    final = torch.zeros(image.shape)

    otf = psf2otf(psf, output_size = image.shape[1:3])
    h = torch.div(torch.conj(otf), (torch.abs(otf) ** 2 + K))

    for i in range(0,3): # iterate over RGB color channels
        channel = torch.Tensor(image[i,:,:])

        img_complex = stack_complex(torch.real(channel), torch.imag(channel))
        img_fft = torch.fft(img_complex, signal_ndim=2)

        filtered = mul_complex(img_fft, h)
        filtered = torch.ifft(filtered, signal_ndim=2)
        filtered = filtered[:,:,0]
        final[i,:,:] = torch.clamp(filtered, min=0, max=255)
    return final

def wiener_filter(image, kernel, K):
    image = image.numpy()
    kernel = kernel.numpy()
    dummy = np.copy(image)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s=image.shape)
    kernel = np.conj(kernel)/(np.abs(kernel)**2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy

class NoisySBDataset():
    def __init__(self,
                 data_root,
                 hyps):
        super().__init__()

        self.psf = hyps['psf_file']
        self.transforms = Compose([
            CenterCrop(size=(256,256)),
            Resize(size=(512,512)),
            ToTensor()
        ])
        self.K = hyps['K']

        # if you set download=True AND you've downloaded the files,
        # it'll never finish running :-(
        self.dataset = torchvision.datasets.SBDataset(root=data_root,
                                                   image_set=hyps['train_test'],
                                                   download=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx): # a[x] for calling a.__getitem__(x)
        '''Returns tuple of (model_input, ground_truth)
        Modifies each item of the dataset upon retrieval
        Convolves with specified PSF
        '''
        img, _ = self.dataset[idx]
        if self.transforms:
            img = self.transforms(img)
        img = torch.Tensor(srgb_to_linear(img))
        psf = torch.Tensor(np.load(self.psf))
        psf /= psf.sum()

        blurred_img = convolve_img(img, psf)

        final = torch.zeros(blurred_img.shape)
        for i in range(0,3):
            channel = blurred_img[i,:,:]
            channel = wiener_filter(channel, psf, K=self.K)
            final[i,:,:] = torch.Tensor(channel)

        #blurred_img = wiener_filter(blurred_img, psf, K=0.001)
        # makes image darker
        #blurred_img = torch.Tensor(linear_to_srgb(blurred_img))
        blurred_img = final
        return blurred_img, img

class NoisyCIFAR10Dataset():
    '''Dataset class that adds noise to cifar10-images'''
    def __init__(self,
                 data_root,
                 hyps):
        super().__init__()

        self.psf = hyps['psf_file']
        self.transforms = Compose([
            CenterCrop(size=(128,128)),
            Resize(size=(256,256)),
            ToTensor()
        ])
        self.dataset = torchvision.datasets.CelebA(root=data_root,
                                                   split='train',
                                                   download=False,
                                                   transform=self.transforms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx): # a[x] for calling a.__getitem__(x)
        '''Returns tuple of (model_input, ground_truth)
        Modifies each item of the dataset upon retrieval
        Convolves with specified PSF
        '''
        img, _ = self.dataset[idx]
        img = torch.Tensor(srgb_to_linear(img))
        psf = torch.Tensor(np.load(self.psf))

        blurred_img = convolve_img(img, psf)
        blurred_img = torch.Tensor(linear_to_srgb(blurred_img))
        return blurred_img, img
