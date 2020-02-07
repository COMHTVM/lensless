import torch
from torch.utils import data
import torchvision
from torchvision.transforms import *
import matplotlib.pyplot as plt
import sys
import torch.nn as nn

sys.path.append("../")
from utils import *

def srgb_to_linear(img):
    return np.where(img <= 0.0031308, 12.92 * img, 1.055 * img ** (0.41666) - 0.055)

def linear_to_srgb(img):
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def psf2otf(input_filter, output_size):
    '''
    Convert pytorch tensor filter into its FFT
    Input:
        input_filter: tensor (height, width)
        output_size: (height, width, num_channels) of image
    Output:
        oft of size output size
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
    Parameters:
        image: (num_channels, height, width)
        psf: (height, width)
    Output:
        final: convolved image (num_channels, height, width)
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
        self.dataset = torchvision.datasets.SBDataset(root=data_root,
                                                   image_set='train',
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

        blurred_img = convolve_img(img, psf)
        blurred_img = inverse_filter(blurred_img, psf, K=0.1)
        blurred_img = torch.Tensor(linear_to_srgb(blurred_img))
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
                                                   download=True,
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
