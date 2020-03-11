import os
import json
from dataio import *
from torch.utils.data import DataLoader
from denoising_unet import DenoisingUnet
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
import torch.nn
import time
from queue import Queue

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


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
    loader = transforms.Compose([transforms.CenterCrop(size=(512,512)),
                                 transforms.ToTensor()])

    image = Image.open(img_name)
    image = loader(image).float().cpu()
    image = torch.Tensor(optics.srgb_to_linear(image))
    blurred_image = image.unsqueeze(0)  # specify a batch size of 1
    image = image.unsqueeze(0)
    return blurred_image.to(DEVICE), image.to(DEVICE)


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

    # *** load model and data set ****
    model = DenoisingUnet(hyps=hyps)

    if not hyps['single_image']:
        dataset = NoisySBDataset(hyps=hyps)
        dataloader = DataLoader(dataset, batch_size=hyps['batch_size'])
        print('Data loader size: ', len(dataloader))

    if hyps['checkpoint'] is not None:  # if trained model is not given, start new checkpoint
        model.load_state_dict(torch.load(hyps['checkpoint']))

    model.to(DEVICE)

    # *** establish folders for saving experiment ***
    run_init = os.path.join(hyps['logging_root'], hyps['exp_name'])
    os.makedirs(run_init, exist_ok=True)

    file_str = hyps['logging_root'] + '/' + hyps['exp_name']
    hyps['exp_num'] = get_exp_num(file_path=file_str, exp_name=hyps['exp_name'])
    dir_name = "{}/{}_{}".format(hyps['exp_name'], hyps['exp_name'], hyps['exp_num'])
    dir_name += hyps['search_keys']
    print('Saving information to ', dir_name)

    run_dir = os.path.join(hyps['logging_root'], dir_name)

    os.makedirs(run_dir, exist_ok=True)

    # *** set up optimizer and scheduler ***
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=10,
                                                           threshold=1e-4,
                                                           factor=0.1)
    writer = SummaryWriter(run_dir)  # run directory for tensorboard information
    iter = 0

    print('Beginning training...')
    if hyps['single_image']:
        print('Optimizing over a single image...')

    # early stopping criteria
    # TODO: move these params to params.json
    prev_loss = 1000
    stop_count = 0
    tolerance = 1e-4
    early_stop = 800
    epoch_loss = 0

    if hyps['single_image']:  # MINI-LOOP for testing
        model_input, ground_truth = image_loader('data/lamb.png')
        ground_truth = ground_truth.to(DEVICE)
        model_input = model_input.to(DEVICE)

        for epoch in range(hyps['max_epoch']):
            model_outputs = model(model_input)
            optimizer.zero_grad()

            total_loss = model.get_distortion_loss(model_outputs, ground_truth)

            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)

            print("Epoch %03d  total_loss %0.4f" % (epoch, total_loss))

            if not iter:  # on the first iteration
                # Save parameters used into the log directory.
                results_file = run_dir + "/params.txt"
                with open(results_file, 'a') as f:
                    for k in hyps.keys():
                        f.write(str(k) + ": " + str(hyps[k]) + '\n')
                    f.write("\n")

            iter += 1
            if iter % 10 == 0:
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "heightmap": model.get_heightmap().numpy(),
                    "psf": model.get_psf(hyps),
                    "epoch": epoch,
                    "iter": iter,
                    "hyps": hyps,
                    "loss": total_loss,
                }

                torch.save(save_dict, os.path.join(run_dir, 'model_epoch_%d_iter_%s.pth' % (epoch, iter)))
        results = {"epoch": epoch,
                   "loss": total_loss}
        return results

    for epoch in range(hyps['max_epoch']):
        for model_input, ground_truth in dataloader:

            ground_truth = ground_truth.to(DEVICE)
            model_input = model_input.to(DEVICE)

            model_outputs = model(model_input)
            model.write_updates(writer, model_outputs, ground_truth, model_input, iter, hyps)

            optimizer.zero_grad()

            psnr = model.get_psnr(model_outputs, ground_truth)
            dist_loss = model.get_distortion_loss(model_outputs, ground_truth)
            reg_loss = model.get_regularization_loss(model_outputs, ground_truth)
            total_loss = dist_loss # can include reg_loss in the future
            epoch_loss += total_loss

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
                    for k in hyps.keys():
                        f.write(str(k) + ": " + str(hyps[k]) + '\n')
                    f.write("\n")

            iter += 1
            if iter % 10 == 0:  # used to be 10,000
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "heightmap": model.get_heightmap().numpy(),
                    "psf": model.get_psf(hyps),
                    "epoch": epoch,
                    "iter": iter,
                    "hyps": hyps,
                    "avg_loss": epoch_loss/iter,
                    "loss": total_loss,
                    "psnr": psnr,
                    "K": model.get_damp()
                }

                for k in hyps.keys():
                    if k not in save_dict:
                        save_dict[k] = hyps[k]

                torch.save(save_dict, os.path.join(run_dir, 'model_epoch_%d_iter_%s.pth' % (epoch, iter)))

        if stop_count >= early_stop:
            break
    torch.save(save_dict, os.path.join(run_dir, 'model_epoch_%d_iter_%s.pth' % (epoch, iter)))

    results = {"epoch": epoch,
               "iter": iter,
               "loss": total_loss}
    return results


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

if __name__ == '__main__':
    # *** load params ***
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
    os.makedirs(hyps['logging_root'], exist_ok=True)

    start_time = time.time()
    hyper_search(hyps, ranges)
    print("Total Execution Time: ", time.time() - start_time)
