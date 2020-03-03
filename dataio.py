import torchvision
from torchvision.transforms import *
import sys

sys.path.append("../")
from utils import *


def linear_to_srgb(img):
    return np.where(img <= 0.0031308, 12.92 * img, 1.055 * img ** (0.41666) - 0.055)


def srgb_to_linear(img):
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)


class NoisySBDataset():
    def __init__(self,
                 data_root,
                 hyps):
        super().__init__()

        self.transforms = Compose([
            CenterCrop(size=(256)),
            Resize(size=(512,512)),
            ToTensor()
        ])
        self.K = hyps['K']

        # if you set download=True AND you've downloaded the files,
        # it'll never finish running :-(
        self.dataset = torchvision.datasets.SBDataset(root=data_root,
                                                      image_set=hyps['train_test'],
                                                      download=hyps['download_data'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):  # a[x] for calling a.__getitem__(x)
        '''Returns tuple of (model_input, ground_truth)
        Modifies each item of the dataset upon retrieval
        Convolves with specified PSF
        '''
        img, _ = self.dataset[idx]
        if self.transforms:
            img = self.transforms(img)

        img = torch.Tensor(srgb_to_linear(img))

        return img, img
