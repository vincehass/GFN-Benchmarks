import argparse
import numpy as np
import math
import os
# from tempfile import TemporaryDirectory
from typing import Tuple


import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch.distributions.categorical import Categorical



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, seq_len: int, dropout: float = 0.2):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 2)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken + seq_len + 1)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src = self.embedding(src) 
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


def construct_M(n, b, H, M_size, seed=0):
    np.random.seed(seed) 
    M = []
    for i in range(M_size):
        M.append("".join([np.random.choice(H) for _ in range(n // b)]))
        assert len(M[-1]) == n
    return M

def distance(s1, s2):
    assert len(s1) == len(s2)
    return sum([int(s1[i] != s2[i]) for i in range(len(s1))])

def M_distance(s, M):#@vince Reward R(x) = exp(-min(d(x, x_1)))
    return min([distance(s, ms) for ms in M])

def construct_test_set(M, seed=0):
    np.random.seed(seed) 
    test_set = []
    for s in M:
        test_set.append(s)
        for cnt in range(1, len(s)):
            new_s = list(s)
            subset = np.random.choice(list(range(len(s))), size=cnt, replace=False)
            for i in subset:
                new_s[i] = "0" if s[i] == "1" else "1"
            test_set.append("".join(new_s))
            assert len(test_set[-1]) == len(s)
            assert distance(test_set[-1], s) == cnt
    return test_set

def log_reward(s, M):
    return -M_distance(s, M)

def reward(s, M):
    return np.exp(log_reward(s, M))

def token_seq_to_str(seq, k):
    return "".join([bin(int(v))[2:].zfill(k) for v in seq])

def batch_rewards(batch, M, k):
    batch_np = batch.cpu().numpy()
    rewards = [reward(token_seq_to_str(batch_np[i], k), M) for i in range(batch_np.shape[0])]
    return torch.tensor(rewards)

def batch_log_rewards(batch, M, k):
    batch_np = batch.cpu().numpy()
    log_rewards = [log_reward(token_seq_to_str(batch_np[i], k), M) for i in range(batch_np.shape[0])]
    return torch.tensor(log_rewards)




def process_logits(all_logits, pos_mask, args):
        # Model predicts positional logits p_i and word logits for each position w_ij.
        # The logits used to sample pairs of positions and word (i, j) are computed as p_i + w_ij.
        pos_logits = all_logits[0, :, -(args.n // args.k + 1):] # [batch_size, n/k + 1]
        pos_logits[pos_mask] = float("-inf")
        word_logits = all_logits[:, :, :2**args.k] # [n/k + 1, batch_size, 2^k]
        sum_logits = torch.moveaxis(word_logits, 1, 0) + pos_logits[:, :, None] #[batch_size, n/k + 1, 2^k] @vince(bz, trajectories, states)
        sum_logits = sum_logits.reshape(pos_logits.shape[0], (args.n // args.k + 1) * (2 ** args.k)) #[batch_size, (n/k + 1) * 2^k]
        return pos_logits, word_logits, sum_logits

def sample_forward(sum_logits, sum_uniform, batch, args):
    # There is a bug in pytorch that allows to sample objects that has 0 probability (happens very rarely but still happens).
    # This loop basically resamples until everything is correct.
    while True:
        actions = Categorical(logits=sum_logits.clone()).sample()
        uniform_actions = Categorical(logits=sum_uniform).sample().to(args.device)
        uniform_mask = torch.rand(args.batch_size) < args.rand_action_prob
        actions[uniform_mask] = uniform_actions[uniform_mask]
        positions = actions // (2 ** args.k) #@vince k: param: trade-off (actions, state); action: seq.n; states space consist of seq. n/k
        if (batch[range(args.batch_size), positions] == 2 ** args.k).sum() == args.batch_size:
            break
    assert positions.min() >= 1
    assert positions.max() <= args.n // args.k
    words = actions % (2 ** args.k)
    return actions, positions, words #@vince positions are the states

    

def TB_train_step(model, log_Z, optimizer, Z_optimizer, M, args):
    # This code is pretty simple because all trajectories in our graph have the same length.
    model.train()
    optimizer.zero_grad()
    Z_optimizer.zero_grad()

    # The seqence has length n/k + 1 and at the beginning looks like [2^k + 1, 2^k, 2^k, ..., 2^k].
    # 2^k + 1: [BOS] token, 2^k: token for "empty" word.
    batch = torch.tensor([[2 ** args.k + 1] + ([2 ** args.k] * (args.n // args.k)) for i in range(args.batch_size)]).to(args.device)
    p_forward_sum = torch.zeros(args.batch_size).to(args.device)
    p_backward_sum = torch.zeros(args.batch_size).to(args.device)

    for i in range(args.n // args.k):
        pos_mask = batch != 2 ** args.k
        all_logits = model(batch.T)
        pos_logits, word_logits, sum_logits = process_logits(all_logits, pos_mask, args)

        with torch.no_grad():
            _, _, sum_uniform = process_logits(0.0 * all_logits.clone(), pos_mask, args)

            actions, positions, words = sample_forward(sum_logits, sum_uniform, batch, args)

            batch_cl = batch.clone()
            batch_cl[range(args.batch_size), positions] = words
            batch = batch_cl
 
        p_forward_sum += sum_logits[range(args.batch_size), actions] - torch.logsumexp(sum_logits, dim=-1)
        p_backward_sum += torch.log(torch.tensor(1 / (i + 1))).to(args.device)

    log_rewards = args.reward_exponent * batch_log_rewards(batch[:, 1:], M, args.k).to(args.device).detach()
    loss = (log_Z.sum() + p_forward_sum - p_backward_sum - log_rewards).pow(2).mean() 
    loss.backward()
    optimizer.step()
    Z_optimizer.step()

    assert batch[:, 1:].max() < 2 ** args.k
    return loss.cpu().item(), batch[:, 1:].cpu()