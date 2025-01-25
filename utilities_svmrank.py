import argparse
import datetime
import numpy as np
import gzip
import pickle

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

def rollout_init(jobs_queue, model, limit, flag):

    while not jobs_queue.empty():
        task = jobs_queue.get()
        sample = task['sample']
        seed = task['seed']
        episode = task['episode']
        # print("Collect samples from: {} , seed: {} , {} episode".format(sample, seed, episode))

        with gzip.open(sample, 'rb') as file:
            data = pickle.load(file)
            observation, cands_feats, node_state, tree_state, action, action_set, scores = data['data']
            constraint_features, (edge_indices, edge_features), variable_features = observation
            action_set = action_set.astype(np.int32)

            node_state = np.array(node_state)
            tree_state = np.array(tree_state)

            cands_feats = cands_feats[0]
            if cands_feats.shape[0] == 500:
                cands_feats = cands_feats[action_set]

            best_action = np.where(action_set == action)[0][0]

            variable_features = variable_features[action_set]
            cand_scores = scores[action_set]

        # if len(action_set) >= 10:
        obs = (variable_features, cands_feats, node_state, tree_state)
        model.store_transition(obs, best_action, action_set, cand_scores)
        jobs_queue.task_done()

        if model.memory.nb_entries >= limit:
            flag = True
            return flag

    print('---current has no instances----')
    return flag

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

    # process candidate variable neighborhoods only
    for var, nbr_id, cand_id in zip(*np.intersect1d(nbr_vars, candidates, return_indices=True)):
        cand_states[cand_id, :] = np.concatenate([
            variable_features[var, :],
            nbr_feats[nbr_id].min(axis=0),
            nbr_feats[nbr_id].mean(axis=0),
            nbr_feats[nbr_id].max(axis=0)])

    cand_states[np.isnan(cand_states)] = 0

    return cand_states

def preprocess_variable_features(features, interaction_augmentation, normalization):
    """
    Features preprocessing following Khalil et al. (2016) Learning to Branch in Mixed Integer Programming.

    Parameters
    ----------
    features : 2D np.ndarray
        The candidate variable features to preprocess.
    interaction_augmentation : bool
        Whether to augment features with 2-degree interactions (useful for linear models such as SVMs).
    normalization : bool
        Wether to normalize features in [0, 1] (i.e., query-based normalization).

    Returns
    -------
    variable_features : 2D np.ndarray
        The preprocessed variable features.
    """
    # 2-degree polynomial feature augmentation
    if interaction_augmentation:
        interactions = (
            np.expand_dims(features, axis=-1) * \
            np.expand_dims(features, axis=-2)
        ).reshape((features.shape[0], -1))
        features = np.concatenate([features, interactions], axis=1)

    # query-based normalization in [0, 1]
    if normalization:
        features -= features.min(axis=0, keepdims=True)
        max_val = features.max(axis=0, keepdims=True)
        max_val[max_val == 0] = 1
        features /= max_val

    return features

def load_flat_samples(filename, feat_type, label_type, augment_feats, normalize_feats):
    with gzip.open(filename, 'rb') as file:
        data = pickle.load(file)

    # state, khalil_state, best_cand, cands, cand_scores = sample['data']
    observation, cands_feats, node_state, tree_state, action, action_set, scores = data['data']
    cands_feats = cands_feats[0]
    action_set = action_set.astype(np.int32)
    node_state = np.array(node_state)
    tree_state = np.array(tree_state)
    # constraint_features, (edge_indices, edge_features), variable_features = observation

    # cands = np.array(cands)
    # cand_scores = np.array(cand_scores)

    cand_states = []
    # if feat_type in ('all', 'gcnn_agg'):
    #     cand_states.append(compute_extended_variable_features(observation, action_set))
    if feat_type in ('all', 'khalil'):
        cand_states.append(cands_feats)
    cand_states = np.concatenate(cand_states, axis=1)

    best_cand_idx = np.where(action_set == action)[0][0]

    # feature preprocessing
    cand_states = preprocess_variable_features(cand_states, interaction_augmentation=augment_feats, normalization=normalize_feats)

    if label_type == 'scores':
        cand_labels = scores
    elif label_type == 'bipartite_ranks':
        # scores quantile discretization as in
        # Khalil et al. (2016) Learning to Branch in Mixed Integer Programming
        cand_scores = scores[action_set]
        cand_labels = np.empty(len(cand_scores), dtype=int)
        cand_labels[cand_scores >= 0.8 * cand_scores.max()] = 1
        cand_labels[cand_scores < 0.8 * cand_scores.max()] = 0
    #
    # else:
    #     raise ValueError(f"Invalid label type: '{label_type}'")

    return cand_states, cand_labels, best_cand_idx
