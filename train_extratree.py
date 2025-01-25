import pickle
import os
import argparse
import numpy as np
import pathlib

from sklearn.ensemble import ExtraTreesRegressor
import utilities_tree
from utilities_tree import log, FlatDataset


def load_samples(filenames, size_limit, logfile=None):
  x, y, n = [], [], []
  total_ncands = 0

  data = FlatDataset(filenames)

  for i in range(len(filenames)):
      cand_x, cand_y, _ = data[i]
      ncands = cand_x.shape[0]

      if total_ncands+ncands >= size_limit:
        log(f"  dataset size limit reached ({total_ncands} candidate variables)", logfile)
        break

      x.append(cand_x)
      y.append(cand_y)
      n.append(ncands)
      total_ncands += ncands

      if (i + 1) % 100 == 0:
          log(f"  {i+1}/{len(filenames)} files processed ({total_ncands} candidate variables)", logfile)

  x = np.concatenate(x)
  y = np.concatenate(y)
  n = np.asarray(n)

  return x, y, n


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'problem',
      help='MILP instance type to process.',
      choices=['MIS', 'CA', 'MK']
  )
  parser.add_argument(
      '-s', '--seed',
      help='Random generator seed.',
      type=utilities_tree.valid_seed,
      default=0,
  )
  args = parser.parse_args()
  rng = np.random.RandomState(args.seed)

  # config #
  train_max_size = 250000
  valid_max_size = 100000
  feat_type = 'gcnn_agg'
  feat_qbnorm = False
  feat_augment = False
  label_type = 'scores'

  # input/output directories #
  problem_folders = {
      'MIS': 'mis/500_4',
      'CA': 'ca/100_500',
      'MK': 'mk/100_6',
  }
  problem_folder = problem_folders[args.problem]
  running_dir = f"trained_tree/{args.problem}/{args.seed}"
  os.makedirs(running_dir, exist_ok=True)

  # logging configuration #
  logfile = f"{running_dir}/log.txt"
  log(f"Logfile for extratree on {args.problem} with seed {args.seed}", logfile)

  # data loading #
  train_files = list(pathlib.Path(f'../data/samples/{problem_folder}/train').glob('sample_*.pkl'))
  valid_files = list(pathlib.Path(f'../data/samples/{problem_folder}/valid').glob('sample_*.pkl'))
  log(f"{len(train_files)} training files", logfile)
  log(f"{len(valid_files)} validation files", logfile)

  log("Loading training samples", logfile)
  train_x, train_y, train_ncands = load_samples(rng.permutation(train_files), train_max_size, logfile)
  log(f"  {train_x.shape[0]} training samples", logfile)

  log("Loading validation samples", logfile)
  valid_x, valid_y, valid_ncands = load_samples(valid_files, valid_max_size, logfile)
  log(f"  {valid_x.shape[0]} validation samples", logfile)

  # data normalization
  log("Normalizing datasets", logfile)
  x_shift = train_x.mean(axis=0)
  x_scale = train_x.std(axis=0)
  x_scale[x_scale == 0] = 1
  valid_x = (valid_x - x_shift) / x_scale
  train_x = (train_x - x_shift) / x_scale

  # Saving feature parameters
  with open(f"{running_dir}/feat_specs.pkl", "wb") as file:
      pickle.dump({
          'type': feat_type,
          'augment': feat_augment,
          'qbnorm': feat_qbnorm,
      }, file)

  # save normalization parameters
  with open(f"{running_dir}/normalization.pkl", "wb") as f:
    pickle.dump((x_shift, x_scale), f)

  log("Starting training", logfile)

  # Training
  model = ExtraTreesRegressor(
      n_estimators=100,
      random_state=rng, )
  model.verbose = True
  model.fit(train_x, train_y)
  model.verbose = False

  # Saving model
  with open(f"{running_dir}/model.pkl", "wb") as file:
      pickle.dump(model, file)

  # Testing
  loss = np.mean((model.predict(valid_x) - valid_y) ** 2)
  log(f"Validation RMSE: {np.sqrt(loss):.2f}", logfile)

