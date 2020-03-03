import os
import torch
import json
from dataio import *
import utils
from torch.utils.data import DataLoader
from denoising_unet import DenoisingUnet
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch.nn
import optics
import time
from queue import Queue

device = torch.device('cuda')

def load_json(file_name):
    """ Loads a JSON file into a dictionary
    :param file_name: str - path to json file
    :return: json file loaded into python dict
    """
    file_name = os.path.expanduser(file_name)
    with open(file_name) as f:
        s = f.read()
        j = json.loads(s)
    return j


def image_loader(img_name):
    """ For optimizing over one image (testing)
    Usage: model_input, ground_truth = image_loader('input.png')
    :param img_name: str - path to single image file to load
    :return: Variable tensor of image in the format (1,C,H,W)
    """
    loader = transforms.Compose([transforms.CenterCrop(size=(1250,1250)),
                                 transforms.Resize(size=(2496,2496)),
                                 transforms.ToTensor()])
    loader = transforms.Compose([transforms.CenterCrop(size=(1248,1248)),
                                 transforms.ToTensor()])

    image = Image.open(img_name)
    image = loader(image).float().cpu()
    image = torch.Tensor(srgb_to_linear(image))
    blurred_image = image.unsqueeze(0)  # specify a batch size of 1
    image = image.unsqueeze(0)
    return blurred_image.cuda(), image.cuda()



def get_lr(optimizer):
    """
    :param optimizer: optimizer object
    :return: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_exp_num(file_path, exp_name):
    """
    Find the next open experiment ID number.
    exp_name: str path to the main experiment folder that contains the model folder
    WARNING: don't name experiments with underscores!
    :param file_path: str - path to folder
    :param exp_name: str - name of exp
    :return: e.g. runs/fresnel50/
    """
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


def train(hyps):
    torch.cuda.empty_cache()

    dataset = NoisySBDataset(data_root=hyps['data_root'], hyps=hyps)
    model = DenoisingUnet(img_sidelength=hyps['resolution'], hyps=hyps)


    dataloader = DataLoader(dataset, batch_size=hyps['batch_size'])
    print('Data loader size: ', len(dataloader))

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
    dir_name += hyps['search_keys']
    print('Saving information to ', dir_name)

    run_dir = os.path.join(hyps['logging_root'], dir_name)

    os.makedirs(run_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=5,
                                                           threshold=2e-3,
                                                           factor=0.8)
    writer = SummaryWriter(run_dir)  # run directory for tensorboard information
    iter = 0


    print('Beginning training...')
    if hyps['single_image']:
        print('Optimizing over a single image...')

    # early stopping criteria
    prev_loss = 1000
    stop_count = 0
    tolerance = 1e-2
    early_stop = 100

    for epoch in range(hyps['max_epoch']):
        for model_input, ground_truth in dataloader:
            if hyps['single_image']:
                model_input, ground_truth = image_loader('data/lamb.png')

            ground_truth = ground_truth.cuda()
            model_input = model_input.cuda()

            model_outputs = model(model_input)
            model.write_updates(writer, model_outputs, ground_truth, model_input, iter, hyps)

            optimizer.zero_grad()

            dist_loss = model.get_distortion_loss(model_outputs, ground_truth)
            reg_loss = model.get_regularization_loss(model_outputs, ground_truth)

            total_loss = dist_loss + hyps['reg_weight'] * reg_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)

            print("Iter %07d   Epoch %03d   dist_loss %0.4f reg_loss %0.4f" %
                  (iter, epoch, dist_loss, reg_loss * hyps['reg_weight']))

            writer.add_scalar("scaled_regularization_loss", reg_loss * hyps['reg_weight'], iter)
            writer.add_scalar("distortion_loss", dist_loss, iter)
            writer.add_scalar("learning_rate", get_lr(optimizer), iter)


            if prev_loss - total_loss <= tolerance:
                stop_count += 1
                if stop_count >= early_stop:
                    break
            elif stop_count >= 1:
                stop_count = 0
            prev_loss = total_loss

            if not iter:  # on the first iteration
                # Save parameters used into the log directory.
                results_file = run_dir + "/params.txt"
                with open(results_file, 'a') as f:
                    f.write("Hyperparameters: \n")
                    for k in hyps.keys():
                        f.write(str(k) + ": " + str(hyps[k]) + '\n')
                    f.write("\n")

            iter += 1
            if iter % 100 == 0:  # used to be 10,000
                torch.save(model.state_dict(), os.path.join(run_dir, 'model-epoch_%d_iter_%s.pth' % (epoch, iter)))

        if stop_count >= early_stop:
            break
    torch.save(model.state_dict(), os.path.join(run_dir, 'model-epoch_%d_iter_%s.pth' % (epoch, iter)))

    results = {"epoch": epoch,
               "iter": iter,
               "loss": total_loss}
    return results



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


def fill_hyper_q(hyps, ranges, keys, hyper_q, idx=0):
    """
    Recursive function to fill queue of specified hyperparameter ranges
    :param hyps: dict of hyperparameters
    :param ranges: dict of different hyperparameters to test
    :param keys:
    :param hyper_q: queue of dictionary of hyperparameters
    :param idx: current index of hyperparameter being added
    :return: queue of dictionary of hyperparameters
    """
    if idx >= len(keys):
        hyps['search_keys'] = ""
        for k in keys:
            hyps['search_keys'] += '_' + str(k)+str(hyps[k])
        hyper_q.put({k:v for k,v in hyps.items()})
    else:
        key = keys[idx]
        for param in ranges[key]:
            hyps[key] = param
            hyper_q = fill_hyper_q(hyps, ranges, keys, hyper_q, idx+1)
    return hyper_q


def hyper_search(hyps, ranges):
    """
    Creates a queue of experiments to test (experiment is one set of hyperparameters)
    Saves results
    :param hyps: dictionary of hyperparameters
    :param ranges: dictionary of ranges of hyperparameters to test
    """
    starttime = time.time()

    # make results file
    if not os.path.exists("runs/"+hyps['exp_name']):
        os.mkdir("runs/"+hyps['exp_name'])

    results_file = "runs/"+hyps['exp_name']+"/results.txt"

    with open(results_file,'a') as f:
        f.write("Hyperparameters:\n")
        for k in hyps.keys():
            if k not in ranges:
                f.write(str(k) + ": " + str(hyps[k]) + '\n')
        f.write("\nHyperranges:\n")
        for k in ranges.keys():
            rs = ",".join([str(v) for v in ranges[k]])
            s = str(k) + ": ["+ rs + ']\n'
            f.write(s)
        f.write('\n')

    hyper_q = Queue()
    hyper_q = fill_hyper_q(hyps, ranges, list(ranges.keys()), hyper_q, idx=0)

    print("n_searches:", hyper_q.qsize())

    while not hyper_q.empty():
        print()
        print("Searches left:", hyper_q.qsize(), "-- Running Time:", time.time()-starttime)
        hyps = hyper_q.get()
        results = train(hyps)
        with open(results_file, 'a') as f:
            results = " -- ".join([str(k) + ":" + str(results[k]) \
                                   for k in sorted(results.keys())])
            f.write("\n"+results+"\n")

def main():
    # load params
    params_file = "params.json"
    ranges_file = "ranges.json"
    print()
    print("Using params file:", params_file)
    print("Using ranges files:", ranges_file)
    print()
    hyps = load_json(params_file)
    ranges = load_json(ranges_file)
    hyps_str = ""
    for k, v in hyps.items():
        hyps_str += "{}: {}\n".format(k, v)
    print("Hyperparameters:")
    print(hyps_str)
    print("\nSearching over:")
    print("\n".join(["{}: {}".format(k, v) for k, v in ranges.items()]))

    os.makedirs(hyps['data_root'], exist_ok=True)

    start_time = time.time()
    hyper_search(hyps, ranges)
    print("Total Execution Time: ", time.time() - start_time)


if __name__ == '__main__':
    main()
