from dataclasses import dataclass
import math
from typing import (
    Any,
    List,
    Optional,
    Union,
)
from types import SimpleNamespace as Namespace

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions.categorical import Categorical
from torchsummary import summary
import matplotlib.pyplot as pp
import numpy as np


enu = enumerate
NEG_INF = float("-inf")
Tensor = torch.Tensor
Number = Union[int, float]
State = Any # State from env file (that is terminal)
Episode = Any # Episode from an env file


def mask_logits(logits: Tensor, mask: Tensor) -> Tensor:
    return torch.where(
        mask, # condition (mask in this case)
        logits,
        torch.tensor(-100.0, dtype=logits.dtype), # set to really small number
    )


def ema(x: List[Number], a=0.10) -> List[Number]:
    res = [x[0]]
    for val in x[1:]:
        xp = ((1-a)*res[-1]) + a*val
        res.append(xp)
    return res


def dither(cat: Categorical, temp=1.0, eps=0.0) -> Categorical:
    # Temper the probs
    probs = torch.softmax(cat.logits / temp, dim=-1)

    # Calc uniform probs
    n_outcomes = cat.probs.size()[0]
    unif_probs = torch.ones(n_outcomes) / n_outcomes

    # Apply uniform probs to tempered probs
    probs = ((1 - eps) * probs) + (eps * unif_probs)
    return Categorical(probs=probs)


class TrajBalMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = 256,
        n_hidden_layers: Optional[int] = 2,
    ):
        super().__init__()
        activation = nn.ReLU

        # logZ
        # - init to 0 (log(1) == 0)
        self.logZ = nn.Parameter(torch.ones(1))

        # Pf, Pb
        self.shared = [nn.Linear(input_dim, hidden_dim), activation()]
        for _ in range(n_hidden_layers - 1):
            self.shared.append(nn.Linear(hidden_dim, hidden_dim))
            self.shared.append(activation())
        self.shared = nn.Sequential(*self.shared)
        self.pf = nn.Linear(hidden_dim, output_dim)
        self.pb = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.shared(x)
        pf = self.pf(x)
        pb = self.pb(x)
        return pf, pb

    def param_sets(self):
        ps = Namespace()
        ps.logZ = [dict(self.named_parameters())["logZ"]]
        ps.policies = [v for k, v in dict(self.named_parameters()).items() if k != "logZ"]
        return ps


@dataclass
class BatchInfo:
    size: int = 0
    steps: int = 0
    loss: float = 0.0
    logZ: float = 0.0
    Pf_entropy: float = 0.0
    Pb_entropy: float = 0.0
    reward: float = 0.0
    max_reward: float = NEG_INF
    grad_norm: float = 0.0


@dataclass
class Batch:
    episodes: List[Any] = None # List[Episode]
    info: BatchInfo = None

    def __post_init__(self):
        self.episodes = []
        self.info = BatchInfo()

    def append(self, episode: Episode):
        self.episodes.append(episode)

    def size(self):
        return len(self.episodes)


@dataclass
class Trainer:
    env: Any
    model: Any = None
    samples: List[State] = None
    batch_info: List[BatchInfo] = None

    def __post_init__(self):
        self.model = None
        self.samples = []
        self.batch_info = []

    def policy_info(self, step, logits, direction) -> Namespace:
        '''
        Calculate relevant policy info from an environment step
        : step - step from Episode.steps()
        : logits - logits of either Pf or Pb
        : direction - "f" if Pf or "b" if Pb
        '''
        assert direction in ("f", "b")
        if direction == "f":
            action = step.action_out
            mask = step.state.f_mask()
        else:
            action = step.action_in
            mask = step.state.b_mask()
        P_valid_action = Categorical(logits=mask_logits(logits, mask))
        action_idx = torch.tensor(self.env.to_action_idx(action)).int()
        return Namespace(
            log_prob=P_valid_action.log_prob(action_idx),
            entropy=P_valid_action.entropy(),
        )

    def process_batch(self, batch: List[Episode]):
        '''
        - Compute trajbal loss
            - sum log of pf/pb for each episode
            - sum log R of each episode
        - Track stats (see BatchInfo)
        '''
        info = batch.info
        for episode in batch.episodes:
            info.size += 1
            log_pf = torch.tensor(0).float()
            log_pb = torch.tensor(0).float()
            final_t = episode.n_steps() - 1
            for step in episode.steps():
                info.steps += 1
                pf_logits, pb_logits = self.model(step.state.encode())

                # Forward probs
                # - Only accumulate for [first, last) state
                if step.t < final_t:
                    pinfo = self.policy_info(step, pf_logits, "f")
                    log_pf += pinfo.log_prob
                    info.Pf_entropy += pinfo.entropy.item()

                # Backward probs
                # - Only accumulate for (first, last] state
                if step.t > 0:
                    pinfo = self.policy_info(step, pb_logits, "b")
                    log_pb += pinfo.log_prob
                    info.Pb_entropy += pinfo.entropy.item()

            # Reward for episode
            # - Accumulate for each episode
            rew = episode.reward()
            info.reward += rew
            info.max_reward = max(rew, info.max_reward)

            # Episode loss
            # XXX: better or worse to have loss divided by batch size?
            logR = torch.tensor(rew).log().clip(-20) # -20 instead of -inf if 0
            ep_loss = (self.model.logZ + log_pf - logR - log_pb).pow(2) / batch.size()
            info.loss += ep_loss

    def train(
        self,
        n_episodes=5000,
        batch_size=16,
        lr_model=3e-4,
        lr_Z=1e-1,
        temp=1.0,
        eps=0.0,
        grad_clip=1e6,
    ):
        # Settings
        device = torch.device("cpu")
        n_batches = n_episodes // batch_size

        # Model
        self.model = TrajBalMLP(
            input_dim=self.env.encoded_state_size(),
            output_dim=self.env.encoded_action_size(),
        )
        self.model.to(device)

        # Train
        params = self.model.param_sets()
        optimizer = torch.optim.Adam([
            {'params': params.policies, 'lr': lr_model},
            {'params': params.logZ, 'lr': lr_Z},
        ])
        for batch_i in (pbar := tqdm(range(n_batches))):
            # Generate Batch
            batch = Batch()
            for _ in range(batch_size):
                episode = self.env.spawn()
                while not episode.done():
                    state = episode.current()
                    pf_logits, _ = self.model(state.encode())
                    P_valid_action = Categorical(logits=mask_logits(pf_logits, state.f_mask()))
                    P_valid_action = dither(P_valid_action, temp=temp, eps=eps)
                    action_index = P_valid_action.sample()
                    action = self.env.to_action(action_index.item())
                    episode.step(action)
                batch.append(episode)
                self.samples.append(episode.current().clone())

            # Update Model
            # - Get target policy Pf/Pb/R quantities
            optimizer.zero_grad()
            self.process_batch(batch)
            loss = batch.info.loss
            loss.backward()
            grad_norm = clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()

            # Monitoring / Report in
            batch.info.grad_norm = grad_norm
            batch.info.logZ = self.model.logZ.item()
            self.batch_info.append(batch.info)
            pbar.set_postfix({
                "loss": loss.item(),
                "z": math.exp(self.model.logZ.item()),
                "maxR": batch.info.max_reward,
            })

    def dashboard(self):
        n_batches = len(self.batch_info)
        batch_size = self.batch_info[0].size

        # xs
        episodes = [x * batch_size for x in range(n_batches)]

        # ys
        loss = [x.loss.item() for x in self.batch_info]
        loss_ema = ema(loss)
        logZ = [x.logZ for x in self.batch_info]
        batch_maxR = [x.max_reward for x in self.batch_info]
        maxR = []
        for i, b_maxR in enu(batch_maxR):
            if i == 0:
                maxR.append(b_maxR)
            else:
                maxR.append(max(maxR[-1], b_maxR))
        avgR = [x.reward / x.size for x in self.batch_info]
        avgR_ema = ema(avgR)
        H_pf = [x.Pf_entropy / x.steps for x in self.batch_info]
        H_pb = [x.Pb_entropy / x.steps for x in self.batch_info]
        grad_norm = [x.grad_norm for x in self.batch_info]
        grad_norm_ema = ema(grad_norm)

        f, ax = pp.subplots(5, 1, figsize=(14, 9))
        pp.sca(ax[0])
        pp.plot(episodes, loss)
        pp.plot(episodes, loss_ema)
        pp.yscale('log')
        pp.ylabel('Loss')
        # ax[0].legend()

        pp.sca(ax[1])
        pp.plot(episodes, np.exp(logZ))
        pp.ylabel('Estimated Z')
        # ax[1].legend()

        pp.sca(ax[2])
        pp.plot(episodes, maxR, label="max(R)")
        pp.plot(episodes, avgR, label="avg(R)")
        pp.plot(episodes, avgR_ema)
        pp.ylabel('Rewards')
        ax[2].legend()

        pp.sca(ax[3])
        pp.plot(episodes, H_pf, label="H[Pf]")
        pp.plot(episodes, H_pb, label="H[Pb]")
        pp.ylabel('Entropy')
        ax[3].legend()

        pp.sca(ax[4])
        pp.plot(episodes, grad_norm)
        pp.plot(episodes, grad_norm_ema)
        pp.ylabel('||Grad||')
        # ax[4].legend()

        pp.xlabel("Batch")
        pp.show()


class Networks:

    @staticmethod
    def summarize(net):
        net.device = "cpu"
        print(net)
        # print(list(net.parameters()))
        summary(net, (1, 10))


class Tasks:

    def inspect_network(self):
        net = TrajBalMLP(input_dim=10, output_dim=5)
        for name, params in net.named_parameters():
            print("\nName:", name)
            print(params)
        Networks.summarize(net)

    def check_dither(self):
        temp = 1.2
        eps = 0.05

        # Build a P_x
        probs = torch.tensor([0.6, 0.3, 0.05, 0.03, 0.02]).float()
        assert abs(probs.sum() - 1.0) <= 0.001
        P_x = Categorical(probs=probs)
        print(P_x, P_x.probs, P_x.logits)

        # Dither it (temper + eps-greedy)
        P_x_temp = dither(P_x, temp=temp)
        P_x_eps = dither(P_x, eps=eps)
        P_x_both = dither(P_x, temp=temp, eps=eps)

        # Display
        print(P_x.probs)
        print(P_x_temp.probs)
        print(P_x_eps.probs)
        print(P_x_both.probs)

    def check_figure(self):
        n_batches = 100
        batch_size = 16
        episodes = [x * batch_size for x in list(range(n_batches))]
        losses = [1000.0 - x for x in range(n_batches)]
        logZs = [x/100 for x in range(n_batches)]

        f, ax = pp.subplots(2, 1, figsize=(10, 6))
        pp.sca(ax[0])
        pp.plot(episodes, losses)
        pp.yscale('log')
        pp.ylabel('loss')
        pp.sca(ax[1])
        pp.plot(episodes, np.exp(logZs))
        pp.ylabel('estimated Z')
        pp.show()


if __name__ == "__main__":
    Tasks().inspect_network()
    Tasks().check_dither()
    Tasks().check_figure()