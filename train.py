import os, datetime
import torch
import json
from dataio import *
from torch.utils.data import DataLoader
from denoising_unet import DenoisingUnet
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
import psutil

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
    params_to_skip = ['data_root','logging_root','batch_size', 'max_epoch', 'train_test']
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
    Usage: model_input = image_loader('input.png')
    ---
    Output: Variable tensor of image of the format (C,H,W)
    '''
    imsize = 200
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

    image = Image.open(img_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0) # specify a batch size of 1
    padder = nn.ZeroPad2d(28)
    image = padder(image)
    print('Image shape: ',image.shape)
    return image.cuda()

def get_exp_num(file_path, exp_name):
    '''
    Find the next open experiment ID number.
    exp_name: str path to the main experiment folder that contains the model folder
    '''
    exp_folder = os.path.expanduser(file_path)
    _,dirs,_ = next(os.walk(exp_folder))
    exp_nums = set()
    for d in dirs:
        splt = d.split("_")
        if len(splt) >=2 and splt[0] == exp_name:
            try:
                print(splt[0])
                exp_nums.add(int(splt[1]))
            except:
                pass
    for i in range(len(exp_nums)):
        if i not in exp_nums:
            return i
    return len(exp_nums)

def train(hyps, model, dataset):
    dataloader = DataLoader(dataset, batch_size=hyps['batch_size'])

    if hyps['checkpoint'] is not None: # if trained model is not given, start new checkpoint
        model.load_state_dict(torch.load(hyps['checkpoint']))

    model.train()
    model.cuda()

    # establish folders for saving experiment
    file_str = hyps['logging_root'] +'/logs'
    hyps['exp_num'] = get_exp_num(file_path=file_str, exp_name=hyps['exp_name'])
    dir_name = "{}_{}".format(hyps['exp_name'],hyps['exp_num'])
    print('Saving information to ', dir_name)
    log_dir = os.path.join(hyps['logging_root'], 'logs', dir_name)
    run_dir = os.path.join(hyps['logging_root'], 'runs', dir_name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'])

    writer = SummaryWriter(run_dir) # run director for tensorboard information
    iter = 0

    writer.add_scalar("learning_rate", hyps['lr'], 0)

    print('Beginning training...')
    for epoch in range(hyps['max_epoch']):
        for model_input, ground_truth in dataloader:
            ground_truth = ground_truth.cuda()
            model_input = model_input.cuda()

            model_outputs = model(model_input)
            model.write_updates(writer, model_outputs, ground_truth, model_input, iter)

            optimizer.zero_grad()

            dist_loss = model.get_distortion_loss(model_outputs, ground_truth)
            reg_loss = model.get_regularization_loss(model_outputs, ground_truth)

            total_loss = dist_loss + hyps['reg_weight'] * reg_loss

            total_loss.backward()
            optimizer.step()

            print("Iter %07d   Epoch %03d   dist_loss %0.4f reg_loss %0.4f" %
                  (iter, epoch, dist_loss, reg_loss * hyps['reg_weight']))

            writer.add_scalar("scaled_regularization_loss", reg_loss * hyps['reg_weight'], iter)
            writer.add_scalar("distortion_loss", dist_loss, iter)

            if not iter: # on the first iteration
                # Save parameters used into the log directory.
                results_file = log_dir + "/params.txt"
                with open(results_file,'a') as f:
                    f.write("Hyperparameters: \n")
                    for k in hyps.keys():
                        f.write(str(k) + ": " + str(hyps[k]) + '\n')
                    f.write("\n")

            iter += 1
            if iter % 10000 == 0:
                torch.save(model.state_dict(), os.path.join(log_dir, 'model-epoch_%d_iter_%s.pth' % (epoch, iter)))

    torch.save(model.state_dict(), os.path.join(log_dir, 'model-epoch_%d_iter_%s.pth' % (epoch, iter)))


def main():
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

    # dataset = NoisyCIFAR10Dataset(data_root=hyps['data_root'], hyps=hyps)
    # model = DenoisingUnet(img_sidelength=32)

    # dataset = DIV2KDataset(data_root=hyps['data_root'])
    # model = DenoisingUnet(img_sidelength=1000)

    dataset = NoisySBDataset(data_root=hyps['data_root'],hyps=hyps)
    model = DenoisingUnet(img_sidelength=512)
    train(hyps, model, dataset)

if __name__ == '__main__':
    main()