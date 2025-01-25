# Generate samples for training, validation and testing.
# e.g., python generate_samples.py MK -s 1 -j 12

import os
import glob
import gzip
import argparse
import pickle
import queue
import shutil
import threading
import numpy as np
import ecole

import math

class obstreeinfo:
    def __init__(self):
        pass

    def before_reset(self, model):
        pass

    def extract(self, model, done=False):
        pyscipopt_model = model.as_pyscipopt()




class ExploreThenStrongBranch:
    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        probabilities = [1-self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model,done), True)
        else:
            return (self.pseudocosts_function.extract(model,done), False)


def send_orders(orders_queue, instances, obj_sence, seed, query_expert_prob, time_limit, out_dir, stop_flag):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : queue.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    stop_flag: threading.Event
        A flag to tell the thread to stop.
    """
    rng = np.random.RandomState(seed)

    episode = 0
    while not stop_flag.is_set():
        instance = rng.choice(instances)
        seed = rng.randint(2**32)
        orders_queue.put([episode, instance, obj_sence, seed, query_expert_prob, time_limit, out_dir])
        episode += 1


def make_samples(in_queue, out_queue, stop_flag):
    """
    Worker loop: fetch an instance, run an episode and record samples.
    Parameters
    ----------
    in_queue : queue.Queue
        Input queue from which orders are received.
    out_queue : queue.Queue
        Output queue in which to send samples.
    stop_flag: threading.Event
        A flag to tell the thread to stop.
    """
    sample_counter = 0
    while not stop_flag.is_set():
        episode, instance, obj_sence, seed, query_expert_prob, time_limit, out_dir = in_queue.get()

        scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0,
                           'limits/time': time_limit, 'timing/clocktype': 2}
        observation_function = {"scores": ExploreThenStrongBranch(expert_probability=query_expert_prob),
                                 "node_observation": ecole.observation.NodeBipartite(),
                                "candidates_feats": ecole.observation.Khalil2016(),
                                "node_info": ecole.observation.FocusNode(),
                                "tree_info": obstreeinfo()
                                }
        env = ecole.environment.Branching(observation_function=observation_function,
                                          scip_params=scip_parameters, pseudo_candidates=True)

        print(f"[w {threading.current_thread().name}] episode {episode}, seed {seed}, "
              f"processing instance '{instance}'...\n", end='')
        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })

        env.seed(seed)

        observation, action_set, a, done, b = env.reset(instance)

        while not done:
            scores, scores_are_expert = observation["scores"]
            node_observation = observation["node_observation"]
            node_observation = (node_observation.row_features,
                                (node_observation.edge_features.indices,
                                 node_observation.edge_features.values),
                                node_observation.column_features)

            cands_obs = observation["candidates_feats"]
            cands_obs = [cands_obs.features]

            focus_node_obs = observation["node_info"]
            lb, ub, depth = focus_node_obs.lowerbound, focus_node_obs.estimate, focus_node_obs.depth
            if obj_sence:
                lb, ub = ub, lb
            node_state = [float(lb), float(ub), float(depth)]

            tree_info = observation["tree_info"]
            info1, info2, info3, info4, info5, info6, lpobjval, gap = tree_info
            if lpobjval != 0:
                reldist = abs(focus_node_obs.depth - lpobjval) / lpobjval
            else:
                reldist = focus_node_obs.depth
            tree_state = [float(reldist), float(info1), float(info2), float(info3), float(info4), float(info5), float(info6)]

            action = action_set[scores[action_set].argmax()]

            if scores_are_expert and not stop_flag.is_set():
                data = [node_observation, cands_obs, node_state, tree_state, action, action_set, scores]
                filename = f'{out_dir}/sample_{episode}_{sample_counter}.pkl'

                with gzip.open(filename, 'wb') as f:
                    pickle.dump({
                        'episode': episode,
                        'instance': instance,
                        'seed': seed,
                        'data': data,
                        }, f)
                out_queue.put({
                    'type': 'sample',
                    'episode': episode,
                    'instance': instance,
                    'seed': seed,
                    'filename': filename,
                })
                sample_counter += 1

            try:
                observation, action_set, a1, done, b1 = env.step(action)
            except Exception as e:
                done = True
                with open("error_log.txt","a") as f:
                    f.write(f"Error occurred solving {instance} with seed {seed}\n")
                    f.write(f"{e}\n")

        print(f"[w {threading.current_thread().name}] episode {episode} done, {sample_counter} samples\n", end='')
        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })


def collect_samples(instances, out_dir, obj_sence, rng, n_samples, n_jobs,
                    query_expert_prob, time_limit):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.
    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """
    os.makedirs(out_dir, exist_ok=True)

    # start workers
    orders_queue = queue.Queue(maxsize=2*n_jobs)
    answers_queue = queue.SimpleQueue()

    tmp_samples_dir = f'{out_dir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # start dispatcher
    dispatcher_stop_flag = threading.Event()
    dispatcher = threading.Thread(
            target=send_orders,
            args=(orders_queue, instances, obj_sence, rng.randint(2**32), query_expert_prob,
                  time_limit, tmp_samples_dir, dispatcher_stop_flag),
            daemon=True)
    dispatcher.start()

    workers = []
    workers_stop_flag = threading.Event()
    for i in range(n_jobs):
        p = threading.Thread(
                target=make_samples,
                args=(orders_queue, answers_queue, workers_stop_flag),
                daemon=True)
        workers.append(p)
        p.start()

    # record answers and write samples
    buffer = {}
    current_episode = 0
    i = 0
    in_buffer = 0
    while i < n_samples:
        sample = answers_queue.get()

        # add received sample to buffer
        if sample['type'] == 'start':
            buffer[sample['episode']] = []
        else:
            buffer[sample['episode']].append(sample)
            if sample['type'] == 'sample':
                in_buffer += 1

        # if any, write samples from current episode
        while current_episode in buffer and buffer[current_episode]:
            samples_to_write = buffer[current_episode]
            buffer[current_episode] = []

            for sample in samples_to_write:

                # if no more samples here, move to next episode
                if sample['type'] == 'done':
                    del buffer[current_episode]
                    current_episode += 1

                # else write sample
                else:
                    os.rename(sample['filename'], f'{out_dir}/sample_{i+1}.pkl')
                    in_buffer -= 1
                    i += 1
                    print(f"[m {threading.current_thread().name}] {i} / {n_samples} samples written, "
                          f"ep {sample['episode']} ({in_buffer} in buffer).\n", end='')

                    # early stop dispatcher
                    if in_buffer + i >= n_samples and dispatcher.is_alive():
                        dispatcher_stop_flag.set()
                        print(f"[m {threading.current_thread().name}] dispatcher stopped...\n", end='')

                    # as soon as enough samples are collected, stop
                    if i == n_samples:
                        buffer = {}
                        break

    # # stop all workers
    workers_stop_flag.set()

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['MIS', 'CA', 'MK']
    )
    parser.add_argument(
        '-s',
        help='Random generator seed.',
        type=int,
        default=0
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=10,
    )
    args = parser.parse_args()

    print(f'collect samples from : {args.problem}')
    print(f"seed {args.seed}")
    print(f"{args.njobs} of parallel jobs.")

    train_size = 150000
    valid_size = 20000
    test_size = 10000
    node_record_prob = 0.05
    time_limit = 3600

    if args.problem == 'MIS':
        maximization = True
        instances_train = glob.glob('data/instances/mis/train_500_4/*.lp')
        instances_valid = glob.glob('data/instances/mis/valid_500_4_all/*.lp')
        instances_test = glob.glob('data/instances/mis/test_500_4/*.lp')
        out_dir = 'data/samples/mis/500_4'

    elif args.problem == 'CA':
        maximization = True
        instances_train = glob.glob('data/instances/ca/train_100_500/*.lp')
        instances_valid = glob.glob('data/instances/ca/valid_100_500/*.lp')
        instances_test = glob.glob('data/instances/ca/test_100_500/*.lp')
        out_dir = 'data/samples/ca/100_500'

    elif args.problem == 'MK':
        maximization = True
        instances_train = glob.glob('data/instances/mk/train_100_6/*.lp')
        instances_valid = glob.glob('data/instances/mk/valid_100_6/*.lp')
        instances_test = glob.glob('data/instances/mk/test_100_6/*.lp')
        out_dir = 'data/samples/mk/100_6'

    else:
        raise NotImplementedError

    print(f"{len(instances_train)} train instances for {train_size} samples")
    print(f"{len(instances_valid)} validation instances for {valid_size} samples")
    print(f"{len(instances_test)} test instances for {test_size} samples")
    obj_sence = maximization

    # create output directory, throws an error if it already exists
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    collect_samples(instances_train, out_dir + '/train', obj_sence, rng, train_size,
                    args.njobs, query_expert_prob=node_record_prob,
                    time_limit=time_limit)

    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances_valid, out_dir + '/valid', obj_sence, rng, valid_size,
                    args.njobs, query_expert_prob=node_record_prob,
                    time_limit=time_limit)

    rng = np.random.RandomState(args.seed + 2)
    collect_samples(instances_test, out_dir + '/test', obj_sence, rng, test_size,
                    args.njobs, query_expert_prob=node_record_prob,
                    time_limit=time_limit)