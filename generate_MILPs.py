# Generates MILPs, i.e., maximum independent set (MIS), combinatorial auction (CA), multiple knapsack (MK)
# e.g., python generate_MILPs.py MIS -s 1

import os
import argparse
import numpy as np
import scipy.sparse
from itertools import combinations


class Graph:
    """
    Container for a graph.

    Parameters
    ----------
    number_of_nodes : int
        The number of nodes in the graph.
    edges : set of tuples (int, int)
        The edges of the graph, where the integers refer to the nodes.
    degrees : numpy array of integers
        The degrees of the nodes in the graph.
    neighbors : dictionary of type {int: set of ints}
        The neighbors of each node in the graph.
    """
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def __len__(self):
        """
        The number of nodes in the graph.
        """
        return self.number_of_nodes

    def greedy_clique_partition(self):
        """
        Partition the graph into cliques using a greedy algorithm.

        Returns
        -------
        list of sets
            The resulting clique partition.
        """
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                # Can you add it to the clique, and maintain cliqueness?
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability, random):
        """
        Generate an Erdös-Rényi random graph with a given edge probability.

        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        edge_probability : float in [0,1]
            The probability of generating each edge.
        random : numpy.random.RandomState
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in combinations(np.arange(number_of_nodes), 2):
            if random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity, random):
        """
        Generate a Barabási-Albert random graph with a given edge probability.

        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        affinity : integer >= 1
            The number of nodes each new node will be attached to, in the sampling scheme.
        random : numpy.random.RandomState
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            # first node is connected to all previous ones (star-shape)
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            # remaining nodes are picked stochastically
            else:
                neighbor_prob = degrees[:new_node] / (2*len(edges))
                neighborhood = random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph


def generate_indset(graph, filename):
    """
    Generate a Maximum Independent Set (also known as Maximum Stable Set) instance
    in CPLEX LP format from a previously generated graph.

    Parameters
    ----------
    graph : Graph
        The graph from which to build the independent set problem.
    filename : str
        Path to the file to save.
    """
    cliques = graph.greedy_clique_partition()
    inequalities = set(graph.edges)
    for clique in cliques:
        clique = tuple(sorted(clique))
        for edge in combinations(clique, 2):
            inequalities.remove(edge)
        if len(clique) > 1:
            inequalities.add(clique)

    # Put trivial inequalities for nodes that didn't appear
    # in the constraints, otherwise SCIP will complain
    used_nodes = set()
    for group in inequalities:
        used_nodes.update(group)
    for node in range(10):
        if node not in used_nodes:
            inequalities.add((node,))

    with open(filename, 'w') as lp_file:
        lp_file.write("maximize\nOBJ:" + "".join([f" + 1 x{node+1}" for node in range(len(graph))]) + "\n")
        lp_file.write("\nsubject to\n")
        for count, group in enumerate(inequalities):
            lp_file.write(f"C{count+1}:" + "".join([f" + x{node+1}" for node in sorted(group)]) + " <= 1\n")
        lp_file.write("\nbinary\n" + " ".join([f"x{node+1}" for node in range(len(graph))]) + "\n")

def generate_cauctions(random, filename, n_items=100, n_bids=500, min_value=1, max_value=100,
                       value_deviation=0.5, add_item_prob=0.9, max_n_sub_bids=5,
                       additivity=0.2, budget_factor=1.5, resale_factor=0.5,
                       integers=False, warnings=False):
    """
    Generate a Combinatorial Auction problem following the 'arbitrary' scheme found in section 4.3. of
        Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
        Towards a universal test suite for combinatorial auction algorithms.
        Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_items : int
        The number of items.
    n_bids : int
        The number of bids.
    min_value : int
        The minimum resale value for an item.
    max_value : int
        The maximum resale value for an item.
    value_deviation : int
        The deviation allowed for each bidder's private value of an item, relative from max_value.
    add_item_prob : float in [0, 1]
        The probability of adding a new item to an existing bundle.
    max_n_sub_bids : int
        The maximum number of substitutable bids per bidder (+1 gives the maximum number of bids per bidder).
    additivity : float
        Additivity parameter for bundle prices. Note that additivity < 0 gives sub-additive bids, while additivity > 0 gives super-additive bids.
    budget_factor : float
        The budget factor for each bidder, relative to their initial bid's price.
    resale_factor : float
        The resale factor for each bidder, relative to their initial bid's resale value.
    integers : logical
        Should bid's prices be integral ?
    warnings : logical
        Should warnings be printed ?
    """

    assert min_value >= 0 and max_value >= min_value
    assert add_item_prob >= 0 and add_item_prob <= 1

    def choose_next_item(bundle_mask, interests, compats, add_item_prob, random):
        n_items = len(interests)
        prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
        prob /= prob.sum()
        return random.choice(n_items, p=prob)

    # common item values (resale price)
    values = min_value + (max_value - min_value) * random.rand(n_items)

    # item compatibilities
    compats = np.triu(random.rand(n_items, n_items), k=1)
    compats = compats + compats.transpose()
    compats = compats / compats.sum(1)

    bids = []
    n_dummy_items = 0

    # create bids, one bidder at a time
    while len(bids) < n_bids:

        # bidder item values (buy price) and interests
        private_interests = random.rand(n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)

        # substitutable bids of this bidder
        bidder_bids = {}

        # generate initial bundle, choose first item according to bidder interests
        prob = private_interests / private_interests.sum()
        item = random.choice(n_items, p=prob)
        bundle_mask = np.full(n_items, 0)
        bundle_mask[item] = 1

        # add additional items, according to bidder interests and item compatibilities
        while random.rand() < add_item_prob:
            # stop when bundle full (no item left)
            if bundle_mask.sum() == n_items:
                break
            item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, random)
            bundle_mask[item] = 1

        bundle = np.nonzero(bundle_mask)[0]

        # compute bundle price with value additivity
        price = private_values[bundle].sum() + np.power(len(bundle), 1 + additivity)
        if integers:
            price = int(price)

        # drop negativaly priced bundles
        if price < 0:
            if warnings:
                print("warning: negatively priced bundle avoided")
            continue

        # bid on initial bundle
        bidder_bids[frozenset(bundle)] = price

        # generate candidates substitutable bundles
        sub_candidates = []
        for item in bundle:

            # at least one item must be shared with initial bundle
            bundle_mask = np.full(n_items, 0)
            bundle_mask[item] = 1

            # add additional items, according to bidder interests and item compatibilities
            while bundle_mask.sum() < len(bundle):
                item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, random)
                bundle_mask[item] = 1

            sub_bundle = np.nonzero(bundle_mask)[0]

            # compute bundle price with value additivity
            sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + additivity)
            if integers:
                sub_price = int(sub_price)

            sub_candidates.append((sub_bundle, sub_price))

        # filter valid candidates, higher priced candidates first
        budget = budget_factor * price
        min_resale_value = resale_factor * values[bundle].sum()
        for bundle, price in [
            sub_candidates[i] for i in np.argsort([-price for bundle, price in sub_candidates])]:

            if len(bidder_bids) >= max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= n_bids:
                break

            if price < 0:
                if warnings:
                    print("warning: negatively priced substitutable bundle avoided")
                continue

            if price > budget:
                if warnings:
                    print("warning: over priced substitutable bundle avoided")
                continue

            if values[bundle].sum() < min_resale_value:
                if warnings:
                    print("warning: substitutable bundle below min resale value avoided")
                continue

            if frozenset(bundle) in bidder_bids:
                if warnings:
                    print("warning: duplicated substitutable bundle avoided")
                continue

            bidder_bids[frozenset(bundle)] = price

        # add XOR constraint if needed (dummy item)
        if len(bidder_bids) > 2:
            dummy_item = [n_items + n_dummy_items]
            n_dummy_items += 1
        else:
            dummy_item = []

        # place bids
        for bundle, price in bidder_bids.items():
            bids.append((list(bundle) + dummy_item, price))

    # generate the LP file
    with open(filename, 'w') as file:
        bids_per_item = [[] for item in range(n_items + n_dummy_items)]

        file.write("maximize\nOBJ:")
        for i, bid in enumerate(bids):
            bundle, price = bid
            file.write(f" +{price} x{i+1}")
            for item in bundle:
                bids_per_item[item].append(i)

        file.write("\n\nsubject to\n")
        for item_bids in bids_per_item:
            if item_bids:
                for i in item_bids:
                    file.write(f" +1 x{i+1}")
                file.write(f" <= 1\n")

        file.write("\nbinary\n")
        for i in range(len(bids)):
            file.write(f" x{i+1}")

def generate_mknapsack(number_of_items, number_of_knapsacks, filename, random,
    min_range=10, max_range=20, scheme='weakly correlated'):
    """
    Generate a Multiple Knapsack problem following a scheme among those found in section 2.1. of
        Fukunaga, Alex S. (2011). A branch-and-bound algorithm for hard multiple knapsack problems.
        Annals of Operations Research (184) 97-119.
    Saves it as a CPLEX LP file.
    Parameters
    ----------
    number_of_items : int
        The number of items.
    number_of_knapsacks : int
        The number of knapsacks.
    filename : str
        Path to the file to save.
    random : numpy.random.RandomState
        A random number generator.
    min_range : int, optional
        The lower range from which to sample the item weights. Default 10.
    max_range : int, optional
        The upper range from which to sample the item weights. Default 20.
    scheme : str, optional
        One of 'uncorrelated', 'weakly correlated', 'strongly corelated', 'subset-sum'. Default 'weakly correlated'.
    """
    weights = random.randint(min_range, max_range, number_of_items)

    if scheme == 'uncorrelated':
        profits = random.randint(min_range, max_range, number_of_items)

    elif scheme == 'weakly correlated':
        profits = np.apply_along_axis(
            lambda x: random.randint(x[0], x[1]),
            axis=0,
            arr=np.vstack([
                np.maximum(weights - (max_range-min_range), 1),
                weights + (max_range-min_range)]))

    elif scheme == 'strongly correlated':
        profits = weights + (max_range - min_range) / 10

    elif scheme == 'subset-sum':
        profits = weights

    else:
        raise NotImplementedError

    capacities = np.zeros(number_of_knapsacks, dtype=int)
    capacities[:-1] = random.randint(0.4 * weights.sum() // number_of_knapsacks,
                                        0.6 * weights.sum() // number_of_knapsacks,
                                        number_of_knapsacks - 1)
    capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

    with open(filename, 'w') as file:
        file.write("maximize\nOBJ:")
        for knapsack in range(number_of_knapsacks):
            for item in range(number_of_items):
                file.write(f" +{profits[item]} x{item+number_of_items*knapsack+1}")

        file.write("\n\nsubject to\n")
        for knapsack in range(number_of_knapsacks):
            variables = "".join([f" +{weights[item]} x{item+number_of_items*knapsack+1}"
                                 for item in range(number_of_items)])
            file.write(f"C{knapsack+1}:" + variables + f" <= {capacities[knapsack]}\n")

        for item in range(number_of_items):
            variables = "".join([f" +1 x{item+number_of_items*knapsack+1}"
                                 for knapsack in range(number_of_knapsacks)])
            file.write(f"C{number_of_knapsacks+item+1}:" + variables + " <= 1\n")

        file.write("\nbinary\n")
        for knapsack in range(number_of_knapsacks):
            for item in range(number_of_items):
                file.write(f" x{item+number_of_items*knapsack+1}")


def valid_seed(seed):
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise argparse.ArgumentTypeError(
                "seed must be any integer between 0 and 2**32 - 1 inclusive")
    return seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['MIS', 'CA', 'MK']
    )
    parser.add_argument(
        '-s',
        help='Random generator seed (default 0).',
        type=valid_seed,
        default=0
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    if args.problem == 'MIS':
        affinity = 4
        number_of_nodes = 500

        filenames = []
        nnodess = []

        # train
        n = 10000
        lp_dir = f'data/instances/mis/train_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # valid
        n = 2000
        lp_dir = f'data/instances/mis/valid_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # test
        n = 100
        lp_dir = f'data/instances/mis/test_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # transfer1
        n = 40
        number_of_nodes = 750
        lp_dir = f'data/instances/mis/transfer_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # transfer2
        n = 40
        number_of_nodes = 1000
        lp_dir = f'data/instances/mis/transfer_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # transfer3
        n = 40
        number_of_nodes = 1250
        lp_dir = f'data/instances/mis/transfer_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # transfer4
        n = 40
        number_of_nodes = 1500
        lp_dir = f'data/instances/mis/transfer_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # actually generate the instances
        for filename, nnodes in zip(filenames, nnodess):
            print(f"  generating file {filename} ...")
            graph = Graph.barabasi_albert(nnodes, affinity, rng)
            generate_indset(graph, filename)

        print("Having done.")

    elif args.problem == 'CA':
        number_of_items = 100
        number_of_bids = 500

        filenames = []
        nitemss = []
        nbidss = []

        # train
        n = 10000
        lp_dir = f'data/instances/ca/train_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids] * n)

        # valid
        n = 2000
        lp_dir = f'data/instances/ca/valid_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids] * n)

        # test instances
        n = 100
        lp_dir = f'data/instances/ca/test_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids] * n)

        # transfer1
        n = 40
        number_of_items = 200
        number_of_bids = 1000
        lp_dir = f'data/instances/ca/transfer_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids] * n)

        # transfer2
        n = 40
        number_of_items = 300
        number_of_bids = 1500
        lp_dir = f'data/instances/ca/transfer_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids] * n)

        # transfer3
        n = 40
        number_of_items = 400
        number_of_bids = 2000
        lp_dir = f'data/instances/ca/transfer_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids] * n)

        # transfer4
        n = 40
        number_of_items = 500
        number_of_bids = 2500
        lp_dir = f'data/instances/ca/transfer_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids] * n)

        # actually generate the instances
        for filename, nitems, nbids in zip(filenames, nitemss, nbidss):
            print(f"  generating file {filename} ...")
            generate_cauctions(rng, filename, n_items=nitems, n_bids=nbids, add_item_prob=0.7)

        print("Having done.")

    elif args.problem == 'MK':
        number_of_items = 100
        number_of_knapsacks = 6

        filenames = []
        nitemss = []
        nknapsackss = []

        # train
        n = 10000
        lp_dir = f'data/instances/mk/train_{number_of_items}_{number_of_knapsacks}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nknapsackss.extend([number_of_knapsacks] * n)

        # valid
        n = 2000
        lp_dir = f'data/instances/mk/valid_{number_of_items}_{number_of_knapsacks}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nknapsackss.extend([number_of_knapsacks] * n)

        # test
        n = 100
        lp_dir = f'data/instances/mk/test_{number_of_items}_{number_of_knapsacks}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nknapsackss.extend([number_of_knapsacks] * n)

        # transfer1
        n = 40
        number_of_items = 100
        number_of_knapsacks = 12
        lp_dir = f'data/instances/mk/transfer_{number_of_items}_{number_of_knapsacks}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nknapsackss.extend([number_of_knapsacks] * n)

        # transfer2
        n = 40
        number_of_items = 200
        number_of_knapsacks = 12
        lp_dir = f'data/instances/mk/transfer_{number_of_items}_{number_of_knapsacks}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nknapsackss.extend([number_of_knapsacks] * n)

        # transfer3
        n = 40
        number_of_items = 250
        number_of_knapsacks = 12
        lp_dir = f'data/instances/mk/transfer_{number_of_items}_{number_of_knapsacks}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nknapsackss.extend([number_of_knapsacks] * n)

        # transfer4
        n = 40
        number_of_items = 350
        number_of_knapsacks = 12
        lp_dir = f'data/instances/mk/transfer_{number_of_items}_{number_of_knapsacks}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nknapsackss.extend([number_of_knapsacks] * n)

        # actually generate the instances
        for filename, nitems, nknapsacks in zip(filenames, nitemss, nknapsackss):
         print(f"  generating file {filename} ...")
         generate_mknapsack(nitems, nknapsacks, filename, rng, min_range=10, max_range=20, scheme='subset-sum')

        print("Having done.")


