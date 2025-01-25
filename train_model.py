import argparse
import glob
import os
import time
import numpy as np
import torch
import utilities
from utilities import log, HyperDataset, load_batch_hyper
from model import Policy

parser = argparse.ArgumentParser()
parser.add_argument(
    'problem',
    help='MILP instance type to process.',
    choices=['MIS', 'CA', 'MK']
)
parser.add_argument(
    '-s',
    help='Random generator seed.',
    type=utilities.valid_seed,
    default=0,
)
parser.add_argument(
    '-g', '--gpu',
    help='CUDA GPU id (-1 for CPU).',
    type=int,
    default=1,
)
args = parser.parse_args()


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

    running_dir = f"trained_models/{args.problem}/{args.seed}"
    os.makedirs(running_dir, exist_ok=True)

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

    max_epochs = 500
    batch_size = 64
    early_stopping = 50
    valid_batch_size = 128
    top_k = [1, 3, 5, 10]
    num_workers = 6
    loss = np.inf
    obj_sence = maximization
    print("{} train samples, {} valid samples".format(len(samples_train), len(samples_valid)))

    # Log
    logfile = os.path.join(running_dir, 'train_log.txt')
    if os.path.exists(logfile):
        os.remove(logfile)
    log(f"max_epochs: {max_epochs}", logfile)
    log(f"train samples : {len(samples_train)}", logfile)
    log(f"valid samples : {len(samples_valid)}", logfile)
    log(f"train_size : {batch_size}", logfile)
    log(f"valid_size : {valid_batch_size}", logfile)
    log(f"early_stopping: {early_stopping}", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed: {args.seed}", logfile)
    log(f"Train_model.", logfile)

    model = Policy(device=device)
    counter = 0

    start_time = time.time()
    start_time_process = time.process_time()
    for epoch in range(max_epochs):
        torch.cuda.empty_cache()
        log(f"current_epoch: {epoch}", logfile)
        epoch_trainsamples = rng.choice(samples_train, 10000, replace=True)
        train_data = HyperDataset(epoch_trainsamples)
        train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers, collate_fn=load_batch_hyper)

        # train
        model.train()
        train_loss, train_kacc = model.update(batch_size=batch_size, dataloader=train_data)
        log(f"TRAIN epoch_loss: {train_loss:0.3f} " + "".join(
                    [f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

        # valid
        valid_data = HyperDataset(samples_valid)
        valid_data = torch.utils.data.DataLoader(valid_data, batch_size=valid_batch_size,
                                                 shuffle=False, num_workers=num_workers, collate_fn=load_batch_hyper)
        model.eval()
        valid_loss, valid_kacc = model.eval_hy(batch_size=valid_batch_size, dataloader=valid_data)
        log(f"VALID epoch_loss: {valid_loss:0.3f} " + "".join(
                [f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

        if valid_loss <= loss:
            counter = 0
            loss = valid_loss
            log(f"Best loss so far.", logfile)
            model.net.save_state(os.path.join(running_dir, f'best_params.pkl'))
        else:
            counter += 1
            if counter % early_stopping == 0:
                log(f" {counter} epochs without improvement, early stopping", logfile)
                break
            if counter % model.patience == 0:
                log(f"  {counter} epoch_loss without improvement", logfile)
                model.lr *= 0.2
                log(f" learning rate is {model.lr}", logfile)
        model.scheduler.step(valid_loss)


    finish_time = time.time() - start_time
    finish_start_time_process = time.process_time() - start_time_process
    print('---time--all: {}, {}'.format(finish_time, finish_start_time_process))