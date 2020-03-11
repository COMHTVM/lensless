import skimage.measure
import torchvision
import utils
from pytorch_prototyping.pytorch_prototyping import *
import matplotlib.pyplot as plt
import torch.nn
import optics
from propagation import Propagation

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


def num_divisible_by_2(number):
    return np.floor(np.log2(number)).astype(int)


def get_num_net_params(net):
    '''Counts number of trainable parameters in pytorch module'''
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class ConvolveImage(nn.Module):
    def __init__(self, hyps, K, heightmap):
        super(ConvolveImage, self).__init__()
        self.resolution = hyps['resolution']
        self.r_cutoff = hyps['r_cutoff']
        self.wavelength = hyps['wavelength']
        self.focal_length = hyps['focal_length']
        self.pixel_pitch = hyps['pixel_pitch']
        self.refractive_idc = hyps['refractive_idc']
        self.use_wiener = hyps['use_wiener']
        self.heightmap = heightmap
        self.K = K

    def forward(self, x):
        # model point from infinity
        input_field = torch.ones((self.resolution, self.resolution))

        phase_delay = utils.heightmap_to_phase(self.heightmap,
                                               self.wavelength,
                                               self.refractive_idc)

        field = optics.propagate_through_lens(input_field, phase_delay)

        field = optics.circular_aperture(field, self.r_cutoff)

        # kernel_type = 'fresnel_conv' leads to  nans
        element = Propagation(kernel_type='fresnel',
                              propagation_distances=self.focal_length,
                              slm_resolution=[self.resolution, self.resolution],
                              slm_pixel_pitch=[self.pixel_pitch, self.pixel_pitch],
                              wavelength=self.wavelength)

        field = element.forward(field)
        psf = utils.field_to_intensity(field)

        psf /= psf.sum()

        final = optics.convolve_img(x, psf)
        if not self.use_wiener:
            return final.to(DEVICE)
        else:
            # perform Wiener filtering
            final = final.to(DEVICE)
            imag = torch.zeros(final.shape).to(DEVICE)
            img = utils.stack_complex(final, imag)
            img_fft = torch.fft(utils.ifftshift(img), 2)

            otf = optics.psf2otf(psf, output_size=img.shape[2:4])

            otf = torch.stack((otf, otf, otf), 0)
            otf = torch.unsqueeze(otf, 0)
            conj_otf = utils.conj(otf)

            otf_img = utils.mul_complex(img_fft, conj_otf)

            denominator = optics.abs_complex(otf)
            denominator[:, :, :, :, 0] += self.K
            product = utils.div_complex(otf_img, denominator)

            filtered = utils.ifftshift(torch.ifft(product, 2))
            filtered = torch.clamp(filtered, min=1e-5)

            return filtered[:, :, :, :, 0]


# class WienerFilter(nn.Module):
#     """Perform Wiener Filtering with learnable damping factor
#       CUDA backprop issues with module as is
#     """
#
#     def __init__(self, hyps, heightmap, K):
#         super(WienerFilter, self).__init__()
#         self.psf = optics.heightmap_to_psf(hyps, heightmap).to(DEVICE)
#         self.K = K
#
#     def forward(self, x):
#         return optics.wiener_filter(x, self.psf, K=self.K ** 2)


class DenoisingUnet(nn.Module):
    """U-Net-based deconvolution
    Assumes images are scaled from -1 to 1.
    """

    def __init__(self, hyps):
        super().__init__()

        self.norm = nn.InstanceNorm2d
        self.img_sidelength = hyps['resolution']

        num_downs_unet = num_divisible_by_2(512)

        self.nf0 = 64  # Number of features to use in the outermost layer of U-Net

        init_heightmap = optics.heightmap_initializer(focal_length=hyps['focal_length'],
                                                      resolution=hyps['resolution'],
                                                      pixel_pitch=hyps['pixel_pitch'],
                                                      refractive_idc=hyps['refractive_idc'],
                                                      wavelength=hyps['wavelength'],
                                                      init_lens=hyps['init_lens'])

        self.heightmap = nn.Parameter(init_heightmap, requires_grad=True)
        self.K = nn.Parameter(torch.ones(1) * hyps['init_K'])

        torch.random.manual_seed(0)

        modules = []

        modules.append(ConvolveImage(hyps,
                                     K=self.K,
                                     heightmap=self.heightmap))

        # TODO: implement wiener filtering as a separate module
        # if hyps['learn_wiener']:
        #     modules.append(WienerFilter(psf, K=self.K))
        # else:
        #     modules.append(WienerFilter(psf, K=hyps['K']))

        # if hyps["use_wiener"]:
        #     modules.append(WienerFilter(hyps, heightmap=self.heightmap, K=self.K))

        # modules.append(Unet(in_channels=3,
        #                     out_channels=3,
        #                     use_dropout=False,
        #                     nf0=self.nf0,
        #                     max_channels=8 * self.nf0,
        #                     norm=self.norm,
        #                     num_down=num_downs_unet,
        #                     outermost_linear=True))
        # modules.append(nn.Tanh())

        self.denoising_net = nn.Sequential(*modules)

        # Losses
        self.loss = nn.MSELoss()

        # List of logs
        self.counter = 0  # A counter to enable logging every nth iteration
        self.logs = list()
        self.learned_gamma = list()

        self.to(DEVICE)

        # print("*" * 100)
        # print(self)  # Prints the model
        # print("*" * 100)
        print("Number of parameters: %d" % get_num_net_params(self))
        print("*" * 100)

    def get_distortion_loss(self, prediction, ground_truth):
        trgt_imgs = ground_truth.to(DEVICE)

        return self.loss(prediction, trgt_imgs)

    def get_regularization_loss(self, prediction, ground_truth):
        return torch.Tensor([0]).to(DEVICE)

    def write_updates(self, writer, predictions, ground_truth, input, iter, hyps):
        """Writes out tensorboard scalar and figures."""
        batch_size, _, _, _ = predictions.shape
        ground_truth = ground_truth.to(DEVICE)

        output_input_gt = torch.cat((predictions, ground_truth), dim=0)
        grid = torchvision.utils.make_grid(output_input_gt,
                                           scale_each=True,
                                           nrow=batch_size,
                                           normalize=True).cpu().detach().numpy()
        writer.add_image("Output_vs_gt", grid, iter)

        writer.add_scalar("psnr", self.get_psnr(predictions, ground_truth), iter)
        writer.add_scalar("damp", self.get_damp(), iter)
        writer.add_figure("heightmap", self.get_heightmap_fig(), iter)

        psf = self.get_psf(hyps)
        plt.figure()
        plt.imshow(psf)
        plt.colorbar()
        fig = plt.gcf()
        plt.close()
        writer.add_figure("psf", fig, iter)

    def get_psnr(self, predictions, ground_truth):
        """Calculates the PSNR of the model's prediction."""
        batch_size, _, _, _ = predictions.shape
        pred = predictions.detach().cpu().numpy()
        gt = ground_truth.detach().cpu().numpy()

        return skimage.measure.compare_psnr(gt, pred, data_range=2)

    def get_damp(self):
        """Returns current Wiener filtering damping factor."""
        return self.K.data.cpu()

    def get_psf(self, hyps):
        """Returns the PSF of the current heightmap."""
        psf = optics.heightmap_to_psf(hyps, self.get_heightmap())
        return psf.cpu().numpy()

    def get_heightmap_fig(self):
        """Wrapper function for getting heightmap and returning
        figure handle."""
        x = self.heightmap.data.cpu().numpy()
        plt.figure()
        plt.imshow(x)
        plt.colorbar()
        fig = plt.gcf()
        return fig

    def get_heightmap(self):
        """Returns heightmap."""
        return self.heightmap.data.cpu()

    def forward(self, input):
        self.logs = list()  # Resets the logs
        return self.denoising_net(input)
