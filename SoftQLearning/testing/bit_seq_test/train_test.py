import argparse
import numpy as np
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
from scipy.stats import spearmanr

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch.distributions.categorical import Categorical

from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tensordict import TensorDict

from utils import construct_M, construct_test_set, TransformerModel, batch_rewards

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument("--n", default=32, type=int)
parser.add_argument("--k", default=4, type=int)
parser.add_argument("--M_size", default=60, type=int)
parser.add_argument("--mode_threshold", default=30, type=int)
parser.add_argument("--reward_exponent", default=2.0, type=float)
parser.add_argument("--seed", default=0, type=int)

# parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--num_iterations", default=1000, type=int)
parser.add_argument("--rand_action_prob", default=0.001, type=float)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--print_modes", default=False, action='store_true')

parser.add_argument("--objective", default='tb', type=str)
parser.add_argument("--z_learning_rate", default=0.001, type=float)
parser.add_argument("--subtb_lambda", default=1.9, type=float)
parser.add_argument("--leaf_coeff", default=5.0, type=float)
parser.add_argument("--update_target_every", default=5, type=int)
parser.add_argument("--corr_num_rounds", default=10, type=int)

# Replay buffer parameters
parser.add_argument("--rb_size", default=100_000, type=int)
parser.add_argument("--rb_batch_size", default=256, type=int)
parser.add_argument("--per_alpha", default=0.9, type=float)
parser.add_argument("--per_beta", default=0.1, type=float)
parser.add_argument("--anneal_per_beta", default=False, action='store_true')




def TB_train_test(model, log_Z, optimizer, Z_optimizer, M, args):
    # This code is pretty simple because all trajectories in our graph have the same length.
    model.train()
    optimizer.zero_grad()
    Z_optimizer.zero_grad()

    # The seqence has length n/k + 1 (32/4 + 1 = 9) and at the beginning looks like [2^k + 1, 2^k, 2^k, ..., 2^k].
    # 2^k + 1: [BOS] token, 2^k: token for "empty" word.
    batch = torch.tensor([[2 ** args.k + 1] + ([2 ** args.k] * (args.n // args.k)) for i in range(args.batch_size)]).to(args.device)
    p_forward_sum = torch.zeros(args.batch_size).to(args.device)
    p_backward_sum = torch.zeros(args.batch_size).to(args.device)

    # for i in range(args.n // args.k):
    #     pos_mask = batch != 2 ** args.k
    #     all_logits = model(batch.T)
    #     pos_logits, word_logits, sum_logits = process_logits(all_logits, pos_mask, args)

    #     with torch.no_grad():
    #         _, _, sum_uniform = process_logits(0.0 * all_logits.clone(), pos_mask, args)

    #         actions, positions, words = sample_forward(sum_logits, sum_uniform, batch, args)

    #         batch_cl = batch.clone()
    #         batch_cl[range(args.batch_size), positions] = words
    #         batch = batch_cl
 
    #     p_forward_sum += sum_logits[range(args.batch_size), actions] - torch.logsumexp(sum_logits, dim=-1)
    #     p_backward_sum += torch.log(torch.tensor(1 / (i + 1))).to(args.device)

    # log_rewards = args.reward_exponent * batch_log_rewards(batch[:, 1:], M, args.k).to(args.device).detach()
    # loss = (log_Z.sum() + p_forward_sum - p_backward_sum - log_rewards).pow(2).mean() 
    # loss.backward()
    # optimizer.step()
    # Z_optimizer.step()

    # assert batch[:, 1:].max() < 2 ** args.k
    # return loss.cpu().item(), batch[:, 1:].cpu()
    return batch, p_forward_sum, p_backward_sum




def main(args):
    
    torch.manual_seed(args.seed)
    device = args.device
    print(f"Device available: {device}")

    assert args.n % args.k == 0
    
    H = ["00000000", "11111111", "11110000", "00001111", "00111100"]
    assert args.n % len(H[0]) == 0
    M = construct_M(args.n, len(H[0]), H, args.M_size, seed=args.seed)
    test_set = construct_test_set(M, seed=args.seed)
    print(f"test set size: {len(test_set)}")

    model = TransformerModel(ntoken=2**args.k+2, d_model=64, d_hid=64, nhead=8, nlayers=3, 
                             seq_len=args.n//args.k, dropout=args.dropout).to(device)
    # if args.objective == "softdqn":
    #     target_model = TransformerModel(ntoken=2**args.k+2, d_model=64, d_hid=64, nhead=8, nlayers=3, 
    #                                     seq_len=args.n//args.k, dropout=args.dropout).to(device)
    #     target_model.load_state_dict(model.state_dict())
        
    log_Z = nn.Parameter(torch.tensor(np.ones(64) * 0.0 / 64, requires_grad=True, device=device))
    
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=1e-5)
    Z_optimizer = torch.optim.Adam([log_Z], args.z_learning_rate, weight_decay=1e-5)

    rb =  TensorDictReplayBuffer(
        storage=LazyTensorStorage(args.rb_size),
        sampler=PrioritizedSampler(
            max_capacity=args.rb_size,
            alpha=args.per_alpha,
            beta=args.per_beta,
        ),
        batch_size=args.rb_batch_size,
        priority_key="td_error"
    )

   
    

    # if args.objective == "softdqn":
    #     # Renormalize entropy for Munchausen DQN
    #     args.entropy_coeff *= 1/(1 - args.m_alpha)
    
    for it in range(args.num_iterations + 1):
        progress = float(it) / args.num_iterations
        if args.objective == "tb":
            p_forward_sum, p_backward_sum, batch = TB_train_test(model, log_Z, optimizer, Z_optimizer, M, args)
        # elif args.objective == "db":
        #     loss, batch = DB_train_step(model, optimizer, M, args)
        # elif args.objective == "subtb":
        #     loss, batch = SubTB_train_step(model, optimizer, M, args)
        # elif args.objective == "softdqn":
        #     # First, collect experiences for experience replay
        #     batch = SoftDQN_collect_experience(rb, model, target_model, M, args)
            # Next, sample transitions from the buffer and calculate the loss
            # if it > args.start_learning:
            #     loss = SoftDQN_learn_rb(progress, rb, model, target_model, optimizer, M, args)
            # else:
            #     loss = 0.0

            # if it % args.update_target_every == 0:
            #     target_model.load_state_dict(model.state_dict())
        
        
        #avg_reward += (batch_rewards(batch, M, args.k) ** args.reward_exponent).sum().item() / args.batch_size
        # rewards_nums.append(avg_reward)
        # batch_strings = [token_seq_to_str(seq, args.k) for seq in batch]
        print(f"iteration: {it}, batch size: {batch.shape}, p_forward_sum: {p_forward_sum.shape}, p_backward_sum: {p_backward_sum.shape}")
    
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
