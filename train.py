import os, datetime
import torch
import json
from dataio import *
import utils
from torch.utils.data import DataLoader
from denoising_unet import DenoisingUnet
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import sys
import numpy as np
import math
from propagation import Propagation
import torch.nn
import optics

device = torch.device('cuda')

def load_json(file_name):
    '''
    Input: file_name str - path to json file
    ---
    Output: json file loaded into python dict
    '''
    file_name = os.path.expanduser(file_name)
    with open(file_name) as f:
        s = f.read()
        j = json.loads(s)
    return j


def params_to_filename(params):
    '''
    Input:
    '''
    params_to_skip = ['data_root', 'logging_root', 'batch_size', 'max_epoch', 'train_test']
    fname = ''
    for key, value in vars(params).items():
        if key in params_to_skip:
            continue
        if key == 'checkpoint' or key == 'data_root' or key == 'logging_root':
            if value is not None:
                value = os.path.basename(os.path.normpath(value))

        fname += "%s_%s_" % (key, value)
    return fname


def image_loader(img_name):
    '''
    Input: img_name str - path to single image file to load
    Usage: model_input, ground_truth = image_loader('input.png')
    ---
    Output: Variable tensor of image of the format (1,C,H,W)
    '''
    loader = transforms.Compose([transforms.CenterCrop(size=(1248,1248)),
                                 transforms.Resize(size=(2496,2496)),
                                 transforms.ToTensor()])

    image = Image.open(img_name)
    image = loader(image).float().cpu()
    image = torch.Tensor(srgb_to_linear(image))
    blurred_image = image.unsqueeze(0)  # specify a batch size of 1
    image = image.unsqueeze(0)
    return blurred_image.cuda(), image.cuda()


def image_loader_blur(img_name,hyps, K):
    '''
    Input: img_name str - path to single image file to load
    Usage: model_input, ground_truth = image_loader('input.png')
    ---
    Output: Variable tensor of image of the format (1,C,H,W)
    '''
    loader = transforms.Compose([transforms.CenterCrop(size=(256, 256)),
                                 transforms.Resize(size=(512, 512)),
                                 transforms.ToTensor()])

    image = Image.open(img_name)
    image = loader(image).float().cpu()
    image = torch.Tensor(srgb_to_linear(image))
    psf = torch.Tensor(np.load(hyps['psf_file']))
    psf /= psf.sum()
    blurred_image = convolve_img(image, psf)

    blurred_rgb = torch.Tensor(linear_to_srgb(torch.squeeze(blurred_image,0)))
    save_image(blurred_rgb, "val/blurred_val.png")

    final = optics.wiener_filter(blurred_image, psf, K=K)
    print('K: ', K)
    final = final.unsqueeze(0)
    print('Image shape: ', final.shape)
    return final.cuda(), image.cuda()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_exp_num(file_path, exp_name):
    '''
    Find the next open experiment ID number.
    exp_name: str path to the main experiment folder that contains the model folder
    ex: runs/fresnel50/
    '''
    exp_folder = os.path.expanduser(file_path)
    _, dirs, _ = next(os.walk(exp_folder))
    exp_nums = set()
    for d in dirs:
        splt = d.split("_")
        if len(splt) >= 2 and splt[0] == exp_name:
            try:
                exp_nums.add(int(splt[1]))
            except:
                pass
    for i in range(len(exp_nums)):
        if i not in exp_nums:
            return i
    return len(exp_nums)


def train(hyps, model, dataset):
    dataloader = DataLoader(dataset, batch_size=hyps['batch_size'])

    if hyps['checkpoint'] is not None:  # if trained model is not given, start new checkpoint
        model.load_state_dict(torch.load(hyps['checkpoint']))

    model.cuda()

    # establish folders for saving experiment
    # make initial logging folders
    run_init = os.path.join(hyps['logging_root'], hyps['exp_name'])
    os.makedirs(run_init, exist_ok=True)

    file_str = hyps['logging_root'] + '/' + hyps['exp_name']
    hyps['exp_num'] = get_exp_num(file_path=file_str, exp_name=hyps['exp_name'])
    dir_name = "{}/{}_{}".format(hyps['exp_name'], hyps['exp_name'], hyps['exp_num'])
    print('Saving information to ', dir_name)
    run_dir = os.path.join(hyps['logging_root'], dir_name)

    os.makedirs(run_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=8,
                                                           threshold=5e-3,
                                                           factor=0.8)
    writer = SummaryWriter(run_dir)  # run directory for tensorboard information
    iter = 0


    print('Beginning training...')
    if hyps['single_image']:
        print('Optimizing over a single image...')
    for epoch in range(hyps['max_epoch']):
        for model_input, ground_truth in dataloader:
            if hyps['single_image']:
                model_input, ground_truth = image_loader('lamb.png')

            ground_truth = ground_truth.cuda()
            model_input = model_input.cuda()

            model_outputs = model(model_input)

            model.write_updates(writer, model_outputs, ground_truth, model_input, iter)

            optimizer.zero_grad()

            psf = optics.heightmap_to_psf(hyps, model.get_heightmap())
            plt.figure()
            plt.imshow(psf)
            plt.colorbar()
            fig = plt.gcf()
            plt.close()

            dist_loss = model.get_distortion_loss(model_outputs, ground_truth)
            reg_loss = model.get_regularization_loss(model_outputs, ground_truth)

            total_loss = dist_loss + hyps['reg_weight'] * reg_loss

            K = model.get_damp()
            heightmap = model.get_heightmap_fig()

            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)

            print("Iter %07d   Epoch %03d   dist_loss %0.4f reg_loss %0.4f" %
                  (iter, epoch, dist_loss, reg_loss * hyps['reg_weight']))

            writer.add_scalar("scaled_regularization_loss", reg_loss * hyps['reg_weight'], iter)
            writer.add_scalar("distortion_loss", dist_loss, iter)
            writer.add_scalar("damp", K, iter)
            writer.add_figure("heightmap", heightmap, iter)
            writer.add_figure("psf", fig, iter)
            writer.add_scalar("learning_rate", get_lr(optimizer), iter)

            if not iter:  # on the first iteration
                # Save parameters used into the log directory.
                results_file = run_dir + "/params.txt"
                with open(results_file, 'a') as f:
                    f.write("Hyperparameters: \n")
                    for k in hyps.keys():
                        f.write(str(k) + ": " + str(hyps[k]) + '\n')
                    f.write("\n")

            iter += 1
            if iter % 3000 == 0:  # used to be 10,000
                torch.save(model.state_dict(), os.path.join(run_dir, 'model-epoch_%d_iter_%s.pth' % (epoch, iter)))

    torch.save(model.state_dict(), os.path.join(run_dir, 'model-epoch_%d_iter_%s.pth' % (epoch, iter)))


def test(hyps, model):
    # load later in training
    epoch = 16
    iter =36000
    hyps['exp_num'] = 0
    dir_name = "{}/{}_{}".format(hyps['exp_name'], hyps['exp_name'], hyps['exp_num'])
    run_dir = os.path.join(hyps['logging_root'], dir_name)
    path_to_model = os.path.join(run_dir, 'model-epoch_%d_iter_%s.pth' % (epoch, iter))
    print('Loading ', path_to_model)

    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    model_input, ground_truth = image_loader_blur('2008_000027_val.jpg', hyps, model.K)

    ground_rgb = torch.Tensor(linear_to_srgb(ground_truth.cpu()))
    save_image(ground_rgb, "val/truth_val.png")
    ground_truth = ground_truth.unsqueeze(0)

    input_rgb = torch.Tensor(linear_to_srgb(model_input.cpu().detach().numpy()))
    save_image(input_rgb, "val/input_val.png")
    with torch.no_grad():
        print()
        print('Testing on single image...')
        model_output = model(model_input)

        output_rgb = torch.Tensor(linear_to_srgb(model_output.cpu()))
        save_image(output_rgb, "val/output_val.png")

        PSNR = model.get_psnr(model_output, ground_truth)
        print('PSNR: ', PSNR)


def main():
    # load params
    params_file = "params.json"
    print()
    print("Using params file:", params_file)
    hyps = load_json(params_file)
    hyps_str = ""
    for k, v in hyps.items():
        hyps_str += "{}: {}\n".format(k, v)
    print("Hyperparameters:")
    print(hyps_str)

    os.makedirs(hyps['data_root'], exist_ok=True)
    dataset = NoisySBDataset(data_root=hyps['data_root'], hyps=hyps)
    model = DenoisingUnet(img_sidelength=512, hyps=hyps)
    #test(hyps, model)
    train(hyps, model, dataset)


if __name__ == '__main__':
    main()
