import torchvision
from torchvision.transforms import *
import sys
import optics
from utils import *


class NoisySBDataset():
    def __init__(self, hyps):
        super().__init__()

        self.transforms = Compose([
            CenterCrop(size=(256,256)),
            Resize(size=(512,512)),
            ToTensor()
        ])

        # if you set download=True AND you've downloaded the files,
        # it'll never finish running :-(
        self.dataset = torchvision.datasets.SBDataset(root=hyps['data_root'],
                                                      image_set=hyps['train_test'],
                                                      download=hyps['download_data'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):  # a[x] for calling a.__getitem__(x)
        """Returns tuple of (model_input, ground_truth)
        Modifies each item of the dataset upon retrieval
        a[x] for calling a.__getitem__(x)
        """
        img, _ = self.dataset[idx]
        if self.transforms:
            img = self.transforms(img)

        img = torch.Tensor(optics.srgb_to_linear(img))

        return img, img