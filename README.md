# A General Hypergraph Decision-making Framework for Mixed Integer Linear Programming

# Generate MILP instances
e.g., python generate_MILPs.py MK -s 1

# Generate supervised learning datasets
python generate_samples.py MK -s 1 -j 12

# Training model
python train_model.py  MK -s 1 -g 1 

# Training baseline models
python train_svmrank.py MK -s 1    
python train_extratree.py MK -s 1
