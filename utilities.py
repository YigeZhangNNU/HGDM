import argparse
import datetime
import torch
import numpy as np
import gzip
import pickle

def _loss_fn(logits, labels):
    loss = torch.nn.CrossEntropyLoss(reduction='none')(logits, labels)
    return torch.sum(loss)

def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

def valid_seed(seed):
    """Check whether seed is a valid random seed or not."""
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise argparse.ArgumentTypeError(
                "seed must be any integer between 0 and 2**32 - 1 inclusive")
    return seed

class HyperDataset(torch.utils.data.Dataset):
    def __init__(self, sample_files):
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        observation, cands_feats, node_state, tree_state, action, action_set, scores = sample['data']
        hyperedge_feats, (hyperedge_indices, edge_features), v_feats = observation

        hyperedge_indices = hyperedge_indices.astype(np.int32)
        action_set = action_set.astype(np.int32)
        node_state = np.array(node_state)
        tree_state = np.array(tree_state)

        return hyperedge_feats, hyperedge_indices, v_feats, node_state, tree_state, action, action_set, scores

