import imageio
import skimage.measure
import torchvision
import utils
from pytorch_prototyping.pytorch_prototyping import *
import matplotlib.pyplot as plt
import torch.nn
import optics


def num_divisible_by_2(number):
    return np.floor(np.log2(number)).astype(int)


def get_num_net_params(net):
    '''Counts number of trainable parameters in pytorch module'''
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class ConvolveImage(nn.Module):
    def __init__(self, hyps, heightmap):
        super(ConvolveImage, self).__init__()
        self.resolution = hyps['resolution']
        self.r_cutoff = hyps['r_cutoff']
        self.wavelength = hyps['wavelength']
        self.focal_length = hyps['focal_length']
        self.pixel_pitch = hyps['pixel_pitch']
        self.refractive_idc = hyps['refractive_idc']
        self.heightmap = heightmap
        self.target_side_length = hyps['target_side_length']

    def forward(self, x):
        input_field = torch.ones((self.resolution, self.resolution))
        saveflag = False

        # heightmap = self.heightmap.pow(2)  # learn the sqrt of the heightmap
        heightmap = self.heightmap
        if saveflag:
            plt.figure()
            plt.imshow(heightmap.data.cpu().numpy())
            plt.colorbar()
            plt.savefig('heightmap_test')
            plt.close()

        phase_delay = utils.heightmap_to_phase(heightmap, self.wavelength, self.refractive_idc)

        field = optics.propagate_through_lens(input_field, phase_delay)

        field = optics.circular_aperture(field, self.r_cutoff)

        field = utils.stack_complex(field, torch.zeros(field.shape))

        # propagate field from aperture to sensor
        field = optics.propagate_fresnel(field,
                                         distance=self.focal_length,
                                         wavelength=self.wavelength,
                                         resolution=self.resolution,
                                         pixel_pitch=self.pixel_pitch)

        psf = utils.field_to_intensity(field)

        if saveflag:
            plt.figure()
            plt.imshow(psf.data.cpu().numpy())
            plt.colorbar()
            plt.savefig('psfcrop_test')
            plt.close()

        psf /= psf.sum()

        final = optics.convolve_img(x, psf)

        if saveflag:
            plt.figure()
            img = final.data.cpu()
            img = torch.squeeze(img, 0)
            img = img.permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.savefig('final_test')
            plt.close()
        return final.cuda()


class WienerFilter(nn.Module):
    ''' Perform Wiener Filtering with learnable damping factor
    if you put in a preloaded psf this works, if you try to
    heightmap_to_psf, it's not on cuda
    '''

    def __init__(self, hyps, heightmap, K):
        super(WienerFilter, self).__init__()
        self.psf = optics.heightmap_to_psf(hyps, heightmap).cuda()
        self.K = K

    def forward(self, x):
        return optics.wiener_filter(x, self.psf, K=self.K ** 2).cuda()


class DenoisingUnet(nn.Module):
    '''A simple unet-based denoiser. This class is overly
    complicated for what it's accomplishing, because it
    serves as an example model class for more complicated models.

    Assumes images are scaled from -1 to 1.
    '''

    def __init__(self,
                 img_sidelength, hyps):
        super().__init__()

        # psf_file = hyps['psf_file']
        # psf = torch.Tensor(np.load(psf_file))
        # psf /= psf.sum()
        # self.psf = psf

        self.norm = nn.InstanceNorm2d
        self.img_sidelength = img_sidelength

        num_downs_unet = num_divisible_by_2(img_sidelength)

        self.nf0 = 64  # Number of features to use in the outermost layer of U-Net

        if hyps['learn_wiener']:
            self.K = nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            self.K = hyps['K']


        init_heightmap = optics.heightmap_initializer(focal_length=hyps['focal_length'],
                                                      resolution=hyps['resolution'],
                                                      pixel_pitch=hyps['pixel_pitch'],
                                                      refractive_idc=hyps['refractive_idc'],
                                                      init_fresnel=hyps['init_fresnel'])
        self.heightmap = nn.Parameter(init_heightmap)

        # torch.random.manual_seed(0)

        modules = []

        modules.append(ConvolveImage(hyps, heightmap=self.heightmap))
        # modules.append(WienerFilter(hyps, heightmap=self.heightmap, K=self.K))
        #
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
        self.loss = nn.MSELoss(reduction='mean')

        # List of logs
        self.counter = 0  # A counter to enable logging every nth iteration
        self.logs = list()
        self.learned_gamma = list()

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.cuda()

        # print("*" * 100)
        # print(self)  # Prints the model
        # print("*" * 100)
        print("Number of parameters: %d" % get_num_net_params(self))
        print("*" * 100)

    def get_distortion_loss(self, prediction, ground_truth):
        trgt_imgs = ground_truth.cuda()

        return self.loss(prediction, trgt_imgs)

    def get_regularization_loss(self, prediction, ground_truth):
        return torch.Tensor([0]).cuda()

    def write_eval(self, prediction, ground_truth, path):
        '''At test time, this saves examples to disk in a format that allows easy inspection'''
        pred = prediction.detach().cpu().numpy()
        gt = ground_truth.detach().cpu().numpy()

        output = np.concatenate((pred, gt), axis=1)
        output /= 2.
        output += 0.5

        imageio.imwrite(path, output)

    def write_updates(self, writer, predictions, ground_truth, input, iter):
        '''Writes out tensorboard summaries as logged in self.logs.'''
        batch_size, _, _, _ = predictions.shape
        ground_truth = ground_truth.cuda()

        output_input_gt = torch.cat((input, predictions, ground_truth), dim=0)
        grid = torchvision.utils.make_grid(output_input_gt,
                                           scale_each=True,
                                           nrow=batch_size,
                                           normalize=True).cpu().detach().numpy()
        writer.add_image("Output_vs_gt", grid, iter)

        writer.add_scalar("psnr", self.get_psnr(predictions, ground_truth), iter)

    def get_psnr(self, predictions, ground_truth):
        '''Calculates the PSNR of the model's prediction.'''
        batch_size, _, _, _ = predictions.shape
        pred = predictions.detach().cpu().numpy()
        gt = ground_truth.detach().cpu().numpy()

        return skimage.measure.compare_psnr(gt, pred, data_range=2)

    def get_damp(self):
        return self.K

    def get_heightmap_fig(self):
        x = self.heightmap.data.cpu().numpy()
        plt.figure()
        plt.imshow(x)
        plt.colorbar()
        fig = plt.gcf()
        return fig

    def get_heightmap(self):
        return self.heightmap.data.cpu()

    def forward(self, input):
        self.logs = list()  # Resets the logs

        batch_size, _, _, _ = input.shape

        noisy_img = input

        pred_noise = self.denoising_net(noisy_img)
        output = pred_noise

        return output
