import argparse
import glob
import os
import numpy as np
import queue
import torch
import utilities_svmrank
from utilities_svmrank import log, load_flat_samples
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    'problem',
    help='MILP instance type to process.',
    choices=['MIS', 'CA', 'MK']
)
parser.add_argument(
    '-s', '--seed',
    help='Random generator seed.',
    type=utilities_svmrank.valid_seed,
    default=0,
)
parser.add_argument(
    '-g', '--gpu',
    help='CUDA GPU id (-1 for CPU).',
    type=int,
    default=1,
)
args = parser.parse_args()

def load_samples(filenames, feat_type, label_type, augment, qbnorm, size_limit, logfile=None):
    x, y, ncands = [], [], []
    total_ncands = 0

    for i, filename in enumerate(filenames):
        cand_x, cand_y, best = load_flat_samples(filename, feat_type, label_type, augment, qbnorm)

        x.append(cand_x)
        y.append(cand_y)
        ncands.append(cand_x.shape[0])
        total_ncands += ncands[-1]

        if (i + 1) % 100 == 0:
            log(f"  {i+1}/{len(filenames)} files processed ({total_ncands} candidate variables)", logfile)

        if total_ncands >= size_limit:
            log(f"  dataset size limit reached ({size_limit} candidate variables)", logfile)
            break

    x = np.concatenate(x)
    y = np.concatenate(y)
    ncands = np.asarray(ncands)

    if total_ncands > size_limit:
        x = x[:size_limit]
        y = y[:size_limit]
        ncands[-1] -= total_ncands - size_limit

    return x, y, ncands

if __name__ == '__main__':

    # Device Setting
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    running_dir = f"trained_svmrank/{args.problem}/{args.seed}"
    os.makedirs(running_dir)

    if args.problem == 'MIS':
        maximization = True
        samples_train = glob.glob('../data/samples/mis/500_4/train/*.pkl')
        samples_valid = glob.glob('../data/samples/mis/500_4/valid/*.pkl')
    elif args.problem == 'CA':
        maximization = True
        samples_train = glob.glob('../data/samples/ca/100_500/train/*.pkl')
        samples_valid = glob.glob('../data/samples/ca/100_500/valid/*.pkl')
    elif args.problem == 'MK':
        maximization = True
        samples_train = glob.glob('../data/samples/mk/100_6/train/*.pkl')
        samples_valid = glob.glob('../data/samples/mk/100_6/valid/*.pkl')
    else:
        raise NotImplementedError

    train_max_size = 250000
    valid_max_size = 100000
    feat_type = 'khalil'
    feat_qbnorm = True
    feat_augment = True
    label_type = 'bipartite_ranks'

    print("{} train samples, {} valid samples".format(len(samples_train), len(samples_valid)))

    # Log
    logfile = os.path.join(running_dir, 'log.txt')
    log(f"train_max_size : {train_max_size}", logfile)
    log(f"valid_max_size : {valid_max_size}", logfile)
    log(f"feat_type: {feat_type}", logfile)
    log(f"feat_qbnorm: {feat_qbnorm}", logfile)
    log(f"feat_augment: {feat_augment}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed: {args.seed}", logfile)
    log(f"label_type: {label_type}", logfile)
    log(f"Train_svmmodel.", logfile)

    log("Loading training samples", logfile)
    train_x, train_y, train_ncands = load_samples(rng.permutation(samples_train), feat_type, label_type,
                                                  feat_augment, feat_qbnorm, train_max_size, logfile)
    log(f"  {train_x.shape[0]} training samples", logfile)

    log("Loading validation samples", logfile)
    valid_x, valid_y, valid_ncands = load_samples(samples_valid, feat_type, label_type,
                                                  feat_augment, feat_qbnorm, valid_max_size, logfile)
    log(f"  {valid_x.shape[0]} validation samples", logfile)

    # Data normalization
    log("Normalizing datasets", logfile)
    x_shift = train_x.mean(axis=0)
    x_scale = train_x.std(axis=0)
    x_scale[x_scale == 0] = 1

    valid_x = (valid_x - x_shift) / x_scale
    train_x = (train_x - x_shift) / x_scale

    log("Starting training", logfile)
    import svmrank

    train_qids = np.repeat(np.arange(len(train_ncands)), train_ncands)
    valid_qids = np.repeat(np.arange(len(valid_ncands)), valid_ncands)

    # Training (includes hyper-parameter tuning)
    best_loss = np.inf
    best_model = None
    for c in (1e-3, 1e-2, 1e-1, 1e0):
        log(f"C: {c}", logfile)
        model = svmrank.Model({
            '-c': c * len(train_ncands),  # c_light = c_rank / n
            '-v': 1,
            '-y': 0,
            '-l': 2,
        })
        model.fit(train_x, train_y, train_qids)
        loss = model.loss(train_y, model(train_x, train_qids), train_qids)
        log(f"  training loss: {loss}", logfile)
        loss = model.loss(valid_y, model(valid_x, valid_qids), valid_qids)
        log(f"  validation loss: {loss}", logfile)
        if loss < best_loss:
            best_model = model
            best_loss = loss
            best_c = c
            # save model
            model.write(f"{running_dir}/model.txt")

    log(f"Best model with C={best_c}, validation loss: {best_loss}", logfile)







