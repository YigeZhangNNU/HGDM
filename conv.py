import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import MessagePassing


class HyperGraphTreeSearch(MessagePassing):

    def __init__(self, emb_size=64, heads=1, training=True, bias=True, negative_slope: float = 0.2, dropout: float = 0.,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.heads = heads
        self.in_channels = emb_size
        self.out_channels = emb_size
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.training = training
        self.bias = bias
        self.concat = True

        self.lin = torch.nn.Linear(self.in_channels, self.heads * self.out_channels, bias=False)
        self.att = torch.nn.Parameter(torch.Tensor(1, self.heads, 2 * self.out_channels))

        self.post = torch.nn.Sequential(
            nn.LayerNorm(self.heads * self.out_channels),
        )

        # output_layer
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * self.heads * emb_size, 2 * emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_size, emb_size),
        )

        if bias and self.concat:
            self.bias = torch.nn.Parameter(torch.Tensor(self.heads * self.out_channels))
        elif bias and not self.concat:
            self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.lin.weight)
        torch.nn.init.orthogonal_(self.att)
        zeros(self.bias)

    def forward(self, x, index, y, d):

        x_i = x[index[0]]
        x_j = y[index[1]]

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        # pass

        out = self.output_module(y)

        return out

    def message(self, x_j, norm_i, alpha):
        pass

class Hypergraph_policy(nn.Module):

    def __init__(self, name='hyper'):
        super(Hypergraph_policy, self).__init__()
        self.name = name
        self.emb_size = 64
        self.hid = 32
        self.var_nfeats = 19
        self.hys_nfeats = 5
        self.milp_state = 10

        self.activation = nn.ReLU()
        self.initializer = lambda x: torch.nn.init.orthogonal_(x, gain=1)

        self.hys_embedding = torch.nn.Sequential(
            nn.LayerNorm(self.hys_nfeats),
            nn.Linear(self.hys_nfeats, self.emb_size, bias=True),
            self.activation,
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            self.activation,
        )

        # VARIABLE_EMBEDDING
        self.var_embedding = nn.Sequential(
            nn.LayerNorm(self.var_nfeats),
            nn.Linear(self.var_nfeats, self.emb_size, bias=True),
            self.activation,
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            self.activation,
        )

        # self.var_embedding2 = nn.Sequential(
        #     nn.LayerNorm(self.var_nfeats),
        #     nn.Linear(self.var_nfeats, self.hid, bias=True),
        #     self.activation,
        #     nn.Linear(self.hid, self.hid, bias=True),
        #     self.activation,
        # )

        # # Tree_EMBEDDING
        self.milp_embedding = nn.Sequential(
            nn.LayerNorm(self.milp_state),
            nn.Linear(self.milp_state, self.emb_size, bias=True),
            self.activation,
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            self.activation,
        )

        # self.milp_embedding = nn.Sequential(
        #     nn.LayerNorm(self.milp_state),
        #     nn.Linear(self.milp_state, self.hid, bias=True),
        #     self.activation,
        #     nn.Linear(self.hid, self.hid, bias=True),
        #     self.activation,
        # )
        #

        # self.embedding = nn.Sequential(
        #     nn.LayerNorm(self.emb_size),
        #     nn.Linear(self.emb_size, self.hid),
        #     self.activation,
        #     nn.Linear(self.hid, self.hid),
        #     self.activation,
        # )

        self.layer1 = HyperGraphTreeSearch(emb_size=self.emb_size, heads=8)
        # self.layer2 = HyperGraphConvolution(emb_size=self.hid, heads=4)
        # self.layers = [self.layer1, self.layer2]


        self.output_module_all = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            self.activation,
            nn.Linear(self.emb_size, 1)
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        for l in self.modules():
            if isinstance(l, torch.nn.Linear):
                self.initializer(l.weight.data)
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias.data, 0)

    @staticmethod
    def pad_output(output, n_vars_per_sample, pad_value=-1e8):
        n_vars_max = torch.max(n_vars_per_sample)

        output = torch.split(
            tensor=output,
            split_size_or_sections=n_vars_per_sample.tolist(),
            dim=1,
        )

        output = torch.cat([
            F.pad(x,
                  pad=[0, n_vars_max - x.shape[1], 0, 0],
                  mode='constant',
                  value=pad_value)
            for x in output
        ], dim=0)

        return output

    def save_state(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore_state(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))

    def forward(self, hyperedge_features, hyperedge_index, hyperedge_weight, variable_features, milp_state):
        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=variable_features.size(0))
        # pass

        B = scatter_add(variable_features.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=hyperedge_features.size(0))
        # pass

        reversed_edge_indices = torch.stack([hyperedge_index[1], hyperedge_index[0]], dim=0)

        v_feats = variable_features = self.var_embedding(variable_features)  # 64
        hyperedge_features = self.hys_embedding(hyperedge_features)
        hyperedge_features = self.layer1(variable_features, hyperedge_index, hyperedge_features, B)

        variable_features = self.layer1(hyperedge_features, reversed_edge_indices, variable_features, D)

        milp_state = self.milp_embedding(milp_state)

        variable_features = torch.split(variable_features, split_size_or_sections=n_vs.tolist(), dim=0)
        variable_features = [tensor_part * milp_state[i:i + 1, :] for i, tensor_part in enumerate(variable_features)]

        variable_features = torch.cat(variable_features, dim=0)
        variable_features = variable_features + v_feats

        output1 = self.output_module_all(variable_features)
        output1 = torch.reshape(output1, [1, -1])
        return output1










