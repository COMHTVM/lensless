import imageio
import skimage.measure
import torchvision
import utils
from propagation import Propagation
from pytorch_prototyping.pytorch_prototyping import *


def num_divisible_by_2(number):
    return np.floor(np.log2(number)).astype(int)


def get_num_net_params(net):
    '''Counts number of trainable parameters in pytorch module'''
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def wiener_filter(img, psf, K):
    ''' Performs Wiener filtering on a single channel
    :param img: pytorch tensor of image (height, width)
    :param psf: pytorch tensor of psf (height, width)
    :param K: damping factor (can be input through hyps or learned)
    :return: Wiener filtered image in one channel (height, width)
    '''
    img = utils.stack_complex(img, torch.zeros(img.shape).cuda())
    img_fft = torch.fft(utils.ifftshift(img), 2).cuda()
    otf = utils.psf2otf(psf, output_size=img.shape[0:2])
    conj_otf = utils.conj(otf).cuda()
    denominator = utils.abs_complex(otf)
    denominator = denominator.cuda()
    denominator[:, :, 0] += K
    k = utils.mul_complex(conj_otf, img_fft)
    g = utils.div_complex(k, denominator)
    filtered = utils.ifftshift(torch.ifft(g, 2))
    filtered = torch.clamp(filtered, min=1e-5)
    return filtered[:, :, 0]


def convolve_img(image, psf):
    ''' Convolves image with a PSF kernel, convolves on each color channel
    :param image: pytorch tensor of image (num_channels, height, width)
    :param psf: pytorch tensor of psf (height, width)
    :return: final convolved image (num_channels, height, width)
    '''
    image = image.cpu()
    final = torch.zeros(image.shape)
    psf = utils.stack_complex(psf, torch.zeros(psf.shape))
    # TODO: get rid of for-loop
    b,_,h,w = image.shape
    for j in range(b):
        single_image = image[j,:,:,:]
        for i in range(0, 3):  # iterate over RGB color channels
            channel = utils.stack_complex(single_image[i,:,:], torch.zeros([h,w]))
            convolved_image = utils.conv_fft(channel, psf, padval=0)
            convolved_image = utils.field_to_intensity(convolved_image)
            final[j,i, :, :] = convolved_image
    return final


def circular_aperture(input_field, r_cutoff):
    ''' Cuts off values outside of aperture area
    Parameters:
        input_field: tensor/array of incoming light field
            (height, width, 2) - last dimension for real and imaginary
        r_cutoff: int in pixels of radius of aperture
    Output:
        filtered: Torch tensor of outgoing light field
    '''
    input_field = input_field.numpy()
    input_shape = input_field.shape
    [x, y] = np.mgrid[-input_shape[0] // 2: input_shape[0] // 2, -input_shape[1] // 2:input_shape[1] // 2].astype(
        np.float64)
    if r_cutoff is None:
        r_cutoff = np.amax(x)
    r = np.sqrt(x ** 2 + y ** 2)
    aperture = (r < r_cutoff)
    real = input_field[:, :, 0] * aperture
    imag = input_field[:, :, 1] * aperture
    filtered = np.stack((real, imag), axis=-1)
    return torch.Tensor(filtered)


def propagate_through_lens(wavelength, curvature, phase_delay):
    wave_nos = 2. * math.pi / wavelength
    wavefront = utils.polar_to_rect(1, wave_nos * curvature)
    wavefront = utils.stack_complex(wavefront[0], wavefront[1])

    phases = utils.polar_to_rect(1, phase_delay)
    phases = utils.stack_complex(phases[0], phases[1])
    return utils.mul_complex(wavefront, phases)


def height_map_initializer(focal_length, resolution, pixel_pitch, wave_length, refractive_idc):
    ''' Models a simple plano convex lens
    '''
    convex_radius = (refractive_idc - 1.) * focal_length  # based on lens maker formula

    N = resolution
    M = resolution
    [x, y] = np.mgrid[-N // 2:N // 2, -M // 2:M // 2].astype(np.float64)

    x = x * pixel_pitch
    y = y * pixel_pitch

    # get lens thickness by paraxial approximations
    height_map = -(x ** 2 + y ** 2) / 2. * (1. / convex_radius)
    return torch.Tensor(height_map)


def propagate_fresnel(field, distance, wavelength, resolution, pixel_pitch):
    pixel_pitch = [pixel_pitch, pixel_pitch]
    resolution = [resolution, resolution]
    element = Propagation(kernel_type='fresnel',
                          propagation_distances=distance,
                          slm_resolution=resolution,
                          slm_pixel_pitch=pixel_pitch,
                          wavelength=wavelength)
    return element.forward(field)


class HeightMap(nn.Module):
    ''' Convolves image with PSF to model point source hitting DOE
    and Fresnel propagate to sensor to get resulting PSF
    '''

    def __init__(self, psf, hyps):
        super(HeightMap, self).__init__()
        self.psf = psf
        self.resolution = hyps['resolution']
        self.r_cutoff = hyps['r_cutoff']
        self.wavelength = hyps['wavelength']
        self.focal_length = hyps['focal_length']
        self.pixel_pitch = hyps['pixel_pitch']
        self.refractive_idc = hyps['refractive_idc']

    def forward(self, x):
        # define input field as planar wave modeled at infinity
        # input_field = torch.ones((self.resolution, self.resolution))
        #
        # height_map = height_map_initializer(self.focal_length,
        #                                     self.resolution,
        #                                     self.pixel_pitch,
        #                                     self.wavelength,
        #                                     self.refractive_idc
        #                                     )
        #
        # phase_delay = utils.heightmap_to_phase(height_map, self.wavelength, self.refractive_idc)
        #
        # field = propagate_through_lens(self.wavelength, input_field, phase_delay)
        #
        # field = circular_aperture(field, self.r_cutoff)
        #
        # # propagate field from aperture to sensor
        # field = propagate_fresnel(field,
        #                           distance=self.focal_length,
        #                           wavelength=self.wavelength,
        #                           resolution=self.resolution,
        #                           pixel_pitch=self.pixel_pitch)
        # psf = utils.field_to_intensity(field)
        # # downsample psf to image resolution (self.patch_size)
        #
        # psf /= psf.sum()
        final = convolve_img(x, self.psf)
        return final.cuda()


class ConvolveImage(nn.Module):
    def __init__(self, psf,hyps):
        super(ConvolveImage, self).__init__()
        self.psf = psf
        self.resolution = hyps['resolution']
        self.r_cutoff = hyps['r_cutoff']
        self.wavelength = hyps['wavelength']
        self.focal_length = hyps['focal_length']
        self.pixel_pitch = hyps['pixel_pitch']
        self.refractive_idc = hyps['refractive_idc']

    def forward(self, x):
        input_field = torch.ones((self.resolution, self.resolution))

        height_map = height_map_initializer(self.focal_length,
                                            self.resolution,
                                            self.pixel_pitch,
                                            self.wavelength,
                                            self.refractive_idc
                                            )
        phase_delay = utils.heightmap_to_phase(height_map, self.wavelength, self.refractive_idc)

        field = propagate_through_lens(self.wavelength, input_field, phase_delay)

        field = circular_aperture(field, self.r_cutoff)

        # propagate field from aperture to sensor
        field = propagate_fresnel(field,
                                  distance=self.focal_length,
                                  wavelength=self.wavelength,
                                  resolution=self.resolution,
                                  pixel_pitch=self.pixel_pitch)
        psf = utils.field_to_intensity(field)
        final = convolve_img(x, psf)
        return final.cuda()


class WienerFilter(nn.Module):
    ''' Perform Wiener Filtering with learnable damping factor
    '''

    def __init__(self, psf, K):
        super(WienerFilter, self).__init__()
        self.psf = psf
        self.K = K

    def forward(self, x):
        # TODO: get rid of for-loop
        final = torch.zeros(x.shape)
        b, _, _, _ = x.shape
        for j in range(b):
            for i in range(0, 3):
                channel = x[j, i, :, :]
                channel = channel.to('cuda:0')
                channel = wiener_filter(channel, self.psf, K=self.K ** 2)
                final[j, i, :, :] = channel
        return final.cuda()


class DenoisingUnet(nn.Module):
    '''A simple unet-based denoiser. This class is overly
    complicated for what it's accomplishing, because it
    serves as an example model class for more complicated models.

    Assumes images are scaled from -1 to 1.
    '''

    def __init__(self,
                 img_sidelength, hyps):
        super().__init__()

        psf_file = hyps['psf_file']
        psf = torch.Tensor(np.load(psf_file))
        psf /= psf.sum()
        self.psf = psf

        self.norm = nn.InstanceNorm2d
        self.img_sidelength = img_sidelength

        num_downs_unet = num_divisible_by_2(img_sidelength)

        self.nf0 = 64  # Number of features to use in the outermost layer of U-Net
        self.K = nn.Parameter(torch.ones(1))

        # torch.random.manual_seed(0)
        # kernel_shape = [hyps['kernel_size'], hyps['kernel_size']]
        # self.height_map = nn.Parameter(torch.rand(kernel_shape))

        modules = []
        # modules.append(HeightMap(psf, hyps))

        # TODO: add modules for light propagation
        # TODO; make heightmap a learnable parameter

        modules.append(ConvolveImage(psf,hyps))
        if hyps['learn_wiener']:
            modules.append(WienerFilter(psf, K=self.K))
        else:
            modules.append(WienerFilter(psf, K=hyps['K']))

        modules.append(Unet(in_channels=3,
                            out_channels=3,
                            use_dropout=False,
                            nf0=self.nf0,
                            max_channels=8 * self.nf0,
                            norm=self.norm,
                            num_down=num_downs_unet,
                            outermost_linear=True))
        modules.append(nn.Tanh())

        self.denoising_net = nn.Sequential(*modules)

        # Losses
        self.loss = nn.MSELoss()

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

        if not self.logs:
            return

        for type, name, content in self.logs:
            if type == 'image':
                writer.add_image(name, content.detach().cpu().numpy(), iter)
                writer.add_scalar(name + '_min', content.min(), iter)
                writer.add_scalar(name + '_max', content.max(), iter)
            elif type == 'figure':
                writer.add_figure(name, content, iter, close=True)

        # Cifar10 images are tiny - to see them better in tensorboard, upsample to 256x256
        output_input_gt = torch.cat((input, predictions, ground_truth), dim=0)
        output_input_gt = F.interpolate(output_input_gt, scale_factor=256 / self.img_sidelength)
        grid = torchvision.utils.make_grid(output_input_gt,
                                           scale_each=True,
                                           nrow=batch_size,
                                           normalize=True).cpu().detach().numpy()
        writer.add_image("Output_vs_gt", grid, iter)

        writer.add_scalar("psnr", self.get_psnr(predictions, ground_truth), iter)

        writer.add_scalar("out_min", predictions.min(), iter)
        writer.add_scalar("out_max", predictions.max(), iter)

        writer.add_scalar("trgt_min", ground_truth.min(), iter)
        writer.add_scalar("trgt_max", ground_truth.max(), iter)

    def get_psnr(self, predictions, ground_truth):
        '''Calculates the PSNR of the model's prediction.'''
        batch_size, _, _, _ = predictions.shape
        pred = predictions.detach().cpu().numpy()
        gt = ground_truth.detach().cpu().numpy()

        return skimage.measure.compare_psnr(gt, pred, data_range=2)

    def get_damp(self):
        return self.K

    def forward(self, input):
        self.logs = list()  # Resets the logs

        batch_size, _, _, _ = input.shape

        noisy_img = input

        # We implement a resnet (good reasoning see https://arxiv.org/abs/1608.03981)
        pred_noise = self.denoising_net(noisy_img)
        output = noisy_img - pred_noise

        if not self.counter % 50:
            # Cifar10 images are tiny - to see them better in tensorboard, upsample to 256x256
            pred_noise = F.interpolate(pred_noise,
                                       scale_factor=256 / self.img_sidelength)
            grid = torchvision.utils.make_grid(pred_noise,
                                               scale_each=True,
                                               normalize=True,
                                               nrow=batch_size)
            self.logs.append(('image', 'pred_noise', grid))

        self.counter += 1

        return output
