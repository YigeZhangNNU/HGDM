import os
import pickle
import pathlib
import argparse
import numpy as np

def svmrank_process(policy, data_loader, top_k=[1, 3, 5, 10]):
  mean_kacc = np.zeros(len(top_k))
  n_samples_processed = 0

  for batch in data_loader:
    feats, best, num_cands = batch
    n_samples = best.shape[0]

    # feature normalization
    feats = (feats - policy['shift']) / policy['scale']

    # prediction
    pred_scores = policy['model'].predict(feats)
    pred_scores = torch.FloatTensor(pred_scores)
    pred_scores = pad_tensor(pred_scores, num_cands)

    # accuracy
    kacc = []
    for k in top_k:
      if pred_scores.size()[-1] < k:
        kacc.append(1.0)
        continue
      pred_top_k = pred_scores.topk(k).indices
      accuracy = (pred_top_k == best).any(dim=-1).float().mean().item()
      kacc.append(accuracy)
    kacc = np.asarray(kacc)

    mean_kacc += kacc * n_samples
    n_samples_processed += n_samples

  mean_kacc /= n_samples_processed
  return mean_kacc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['MIS', 'CA', 'MK']
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random number generator seed.',
        type=int,
        default=1,
    )
    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)

    problem_folders = {
        'MIS': 'mis/500_4',
        'CA': 'ca/100_500',
        'MK': 'mk/100_6',
    }
    problem_folder = problem_folders[args.problem]
    model_dir = f"trained_svmrank/{args.problem}/{args.seed}"

    ## logger setup ##
    logfile = os.path.join(model_dir, 'test_log.txt')
    if os.path.exists(logfile):
        os.remove(logfile)


    import torch
    import svmrank
    from utilities import log, FlatDataset, pad_tensor

    ## load feature normalization parameters ##
    policy = {}
    with open(f"{model_dir}/normalization.pkl", 'rb') as f:
        policy['shift'], policy['scale'] = pickle.load(f)
        print(policy['shift'], policy['scale'])

      ## load data ##
    test_files = [str(file) for file in (pathlib.Path(f'../data/samples')/problem_folder/'test').glob('sample_*.pkl')]
    test_data = FlatDataset(test_files)
    test_loader = torch.utils.data.DataLoader(test_data, 64, shuffle=False, collate_fn=FlatDataset.collate)

    ## load model ##
    policy['model'] = svmrank.Model().read(f"{model_dir}/model.txt")

    ## test ##
    top_k = [1, 3, 5, 10]
    test_kacc = svmrank_process(policy, test_loader)
    log(f"TEST RESULTS: " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, test_kacc)]), logfile)