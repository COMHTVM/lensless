import torchvision
from torchvision.transforms import *
import sys

sys.path.append("../")
from utils import *

def linear_to_srgb(img):
    return np.where(img <= 0.0031308, 12.92 * img, 1.055 * img ** (0.41666) - 0.055)

def srgb_to_linear(img):
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

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


class NoisySBDataset():
    def __init__(self,
                 data_root,
                 hyps):
        super().__init__()

        self.psf_file = hyps['psf_file']
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

        #blurred_img = convolve_img(img, psf)
        blurred_img = img
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

        blurred_img = torch.Tensor(convolve_img(img, psf))
        #blurred_img = torch.Tensor(linear_to_srgb(blurred_img))
        return blurred_img, img
