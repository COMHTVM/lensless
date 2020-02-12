import imageio
import skimage.measure

import torchvision

from pytorch_prototyping.pytorch_prototyping import *


def num_divisible_by_2(number):
    return np.floor(np.log2(number)).astype(int)


def get_num_net_params(net):
    '''Counts number of trainable parameters in pytorch module'''
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class DenoisingUnet(nn.Module):
    '''A simple unet-based denoiser. This class is overly complicated for what it's accomplishing, because it
    serves as an example model class for more complicated models.

    Assumes images are scaled from -1 to 1.
    '''

    def __init__(self,
                 img_sidelength):
        super().__init__()

        self.norm = nn.InstanceNorm2d
        self.img_sidelength = img_sidelength

        num_downs_unet = num_divisible_by_2(img_sidelength)

        self.nf0 = 64  # Number of features to use in the outermost layer of U-Net

        self.denoising_net = nn.Sequential(
            Unet(in_channels=3,
                 out_channels=3,
                 use_dropout=False,
                 nf0=self.nf0,
                 max_channels=8 * self.nf0,
                 norm=self.norm,
                 num_down=num_downs_unet,
                 outermost_linear=True),
            nn.Tanh()
        )

        # Losses
        self.loss = nn.MSELoss()

        # List of logs
        self.counter = 0  # A counter to enable logging every nth iteration
        self.logs = list()

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
