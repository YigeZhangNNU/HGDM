import gzip
import pickle
import datetime
import numpy as np
import argparse

import torch
import torch.nn.functional as F
import torch_geometric

def valid_seed(seed):
  """Check whether seed is a valid random seed or not."""
  seed = int(seed)
  if seed < 0 or seed > 2**32 - 1:
    raise argparse.ArgumentTypeError(
          "seed must be any integer between 0 and 2**32 - 1 inclusive")
  return seed

def log(str, logfile=None):
  str = f'[{datetime.datetime.now()}] {str}'
  print(str)
  if logfile is not None:
    with open(logfile, mode='a') as f:
      print(str, file=f)


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
  max_pad_size = pad_sizes.max()
  output = input_.split(pad_sizes.cpu().numpy().tolist())
  output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                        for slice_ in output], dim=0)
  return output


class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, candidate_choice, candidate_scores):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = len(candidates)
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, _, sample_action, sample_action_set, sample_scores = sample['data']

        constraint_features, (edge_indices, edge_features), variable_features = sample_observation
        constraint_features = torch.FloatTensor(constraint_features)
        edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
        edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
        variable_features = torch.FloatTensor(variable_features)

        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_choice = torch.where(candidates == sample_action)[0][0]  # action index relative to candidates
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                  candidates, candidate_choice, candidate_scores)
        graph.num_nodes = constraint_features.shape[0]+variable_features.shape[0]
        return graph

def compute_extended_variable_features(state, candidates):
    """
    Utility to extract variable features only from a bipartite state representation.

    Parameters
    ----------
    state : dict
        A bipartite state representation.
    candidates: list of ints
        List of candidate variables for which to compute features (given as indexes).

    Returns
    -------
    variable_states : np.array
        The resulting variable states.
    """
    constraint_features, (edge_indices, edge_features), variable_features = state
    edge_indices = edge_indices.astype(np.int32)
    edge_features = np.expand_dims(edge_features, axis=-1)

    cand_states = np.zeros((
        len(candidates),
        variable_features.shape[1] + 3*(edge_features.shape[1] + constraint_features.shape[1]),
    ))

    # re-order edges according to variable index
    edge_ordering = edge_indices[1].argsort()
    edge_indices = edge_indices[:, edge_ordering]
    edge_features = edge_features[edge_ordering]

    # gather (ordered) neighbourhood features
    nbr_feats = np.concatenate([
        edge_features,
        constraint_features[edge_indices[0]]
    ], axis=1)

    # split neighborhood features by variable, along with the corresponding variable
    var_cuts = np.diff(edge_indices[1]).nonzero()[0]+1
    nbr_feats = np.split(nbr_feats, var_cuts)
    nbr_vars = np.split(edge_indices[1], var_cuts)
    assert all([all(vs[0] == vs) for vs in nbr_vars])
    nbr_vars = [vs[0] for vs in nbr_vars]
    # print('nbr_vars:', nbr_vars)

    # process candidate variable neighborhoods only
    for var, nbr_id, cand_id in zip(*np.intersect1d(nbr_vars, candidates, return_indices=True)):
        var = int(var)
        cand_states[cand_id, :] = np.concatenate([
            variable_features[var, :],
            nbr_feats[nbr_id].min(axis=0),
            nbr_feats[nbr_id].mean(axis=0),
            nbr_feats[nbr_id].max(axis=0)])

    cand_states[np.isnan(cand_states)] = float(0)
    cand_states[np.isinf(cand_states)] = float(0)

    return cand_states

class FlatDataset(torch.utils.data.Dataset):
  def __init__(self, filenames):
    self.filenames = filenames

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    with gzip.open(self.filenames[idx], 'rb') as file:
      sample = pickle.load(file)

    state, khalil_state, node_state, tree_state, best_cand, cands, scores = sample['data']

    cands = np.array(cands)
    cand_scores = np.array(scores[cands])

    cand_states = []
    cand_states.append(compute_extended_variable_features(state, cands))
    cand_states = np.concatenate(cand_states, axis=1)

    best_cand_idx = np.where(cands == best_cand)[0][0]

    # scores quantile discretization as in
    cand_labels = np.empty(len(cand_scores), dtype=int)
    cand_labels[cand_scores >= 0.8 * cand_scores.max()] = 1
    cand_labels[cand_scores < 0.8 * cand_scores.max()] = 0

    return cand_states, cand_labels, best_cand_idx

  def collate(batch):
    num_candidates = [item[0].shape[0] for item in batch]
    num_candidates = torch.LongTensor(num_candidates)

    # batch states #
    batched_states = [item[0] for item in batch]
    batched_states = np.concatenate(batched_states, axis=0)
    # batch targets #
    batched_best = [[item[2]] for item in batch]
    batched_best = torch.LongTensor(batched_best)

    return [batched_states, batched_best, num_candidates]

class Scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch =+1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs == self.patience:
            self._reduce_lr(self.last_epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def plot_training_curve(filename):
    import matplotlib.pyplot as plt 
    
    train_loss = []
    valid_loss = []
    train_acc = {1: [], 3: [], 5: [], 10: []}
    valid_acc = {1: [], 3: [], 5: [], 10: []}

    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            line = line.split(' ')

            if line[2] == "TRAIN":
                train_loss.append(float(line[4]))
                train_acc[1].append(float(line[7]))
                train_acc[3].append(float(line[9]))
                train_acc[5].append(float(line[11]))
                train_acc[10].append(float(line[13]))
            if line[2] == "VALID":
                valid_loss.append(float(line[4]))
                valid_acc[1].append(float(line[7]))
                valid_acc[3].append(float(line[9]))
                valid_acc[5].append(float(line[11]))
                valid_acc[10].append(float(line[13]))

    # plot loss #
    plt.plot(train_loss, color='darkslateblue', label='Train')
    plt.plot(valid_loss, color='peru', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()

    # plot accuracy #
    plt.plot(train_acc[5], color='darkslateblue', label='Train')
    plt.plot(valid_acc[5], color='peru', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Acc@5')
    plt.legend()
    plt.show()