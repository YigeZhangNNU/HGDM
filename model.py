import torch
import numpy as np
from hyper_conv import Hypergraph_policy
from utilities import _loss_fn
import torch.nn.functional as F

class Policy(object):
    def __init__(self, device):

        self.lr = 0.001
        self.top_k = [1, 3, 5, 10]
        self.device = device
        self.patience = 10
        self.entropy_bonus = 0.0

        self.net = Hypergraph_policy().to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=self.patience,
                                                               verbose=True)

    def _predict(self, net, hypere_feats, edge_in, hy_w, v_feats, milp_states, n_vs):
        logits = net(hypere_feats, edge_in, hy_w, v_feats, milp_states, n_vs)
        logits = self.net.pad_output(logits, n_vs)
        return logits

    def _act(self, net, hypere_feats, edge_in, hy_w, v_feats, milp_state, n_vs):
        with torch.no_grad():
            logits = net(hypere_feats, edge_in, hy_w, v_feats, milp_state, n_vs)
        return logits

    def choose_action(self, hypere_feats, edge_in, v_feats, milp_state, action_set):
        n_vs = torch.as_tensor(v_feats.size(0), dtype=torch.int32, device=self.device)
        action_set = torch.as_tensor(action_set, dtype=torch.long, device=self.device)
        n_hypere = torch.as_tensor(hypere_feats.size(0), dtype=torch.long, device=self.device)
        hyperedge_index = torch.stack([edge_in[1], edge_in[0]], dim=0)
        hyperedge_weight = v_feats.new_ones(n_hypere)

        logits = self._act(self.net, hypere_feats, hyperedge_index, hyperedge_weight, v_feats, milp_state, n_vs)[action_set]
        max_action_value, action_index = torch.max(logits, dim=0)

        action = action_set[action_index].cpu()
        return action

    def update(self, batch_size, dataloader):
        n_samples = 0
        mean_loss = 0
        mean_kacc = np.zeros(len(self.top_k))

        for batch in dataloader:
            hys, hyperedge_index, hyperedge_weight, var, milp_states, n_vs, best_cands, scores = map(lambda x: x.to(self.device), batch)
            logits = self._predict(self.net, hys, hyperedge_index, hyperedge_weight, var, milp_states, n_vs)
            c_loss = _loss_fn(logits, best_cands)
            entropy = (-F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            loss = c_loss - self.entropy_bonus * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            true_scores = self.net.pad_output(torch.reshape(scores, (1, -1)), n_vs)
            true_bestscore = torch.max(true_scores, dim=-1, keepdims=True).values
            true_scores = true_scores.cpu().numpy()
            true_bestscore = true_bestscore.cpu().numpy()

            kacc = []
            for k in self.top_k:
                pred_top_k = torch.topk(logits, k=k).indices.cpu().numpy()
                pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
                kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1)))
            kacc = np.asarray(kacc)

            mean_loss += loss.detach_().item() * batch_size
            mean_kacc += kacc * batch_size
            n_samples += batch_size
        mean_loss /= n_samples
        mean_kacc /= n_samples
        return mean_loss, mean_kacc

    def eval_hy(self, batch_size, dataloader):
        n_samples = 0
        mean_loss = 0
        mean_kacc = np.zeros(len(self.top_k))

        for batch in dataloader:
            hys, hyperedge_index, hyperedge_weight, var, milp_states, n_vs, best_cands, scores = map(lambda x: x.to(self.device), batch)
            logits = self._act(self.net, hys, hyperedge_index, hyperedge_weight, var, milp_states, n_vs)
            logits = self.net.pad_output(logits, n_vs)
            c_loss = _loss_fn(logits, best_cands)
            entropy = (-F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            loss = c_loss - self.entropy_bonus * entropy

            true_scores = self.net.pad_output(torch.reshape(scores, (1, -1)), n_vs)
            true_bestscore = torch.max(true_scores, dim=-1, keepdims=True).values
            true_scores = true_scores.cpu().numpy()
            true_bestscore = true_bestscore.cpu().numpy()

            kacc = []
            for k in self.top_k:
                pred_top_k = torch.topk(logits, k=k).indices.cpu().numpy()
                pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
                kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1)))
            kacc = np.asarray(kacc)

            mean_loss += loss.detach_().item() * batch_size
            mean_kacc += kacc * batch_size
            n_samples += batch_size

        mean_loss /= n_samples
        mean_kacc /= n_samples
        return mean_loss, mean_kacc

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()