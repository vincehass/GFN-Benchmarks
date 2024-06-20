# Generative Flow Networks as Benchmark

## Soft Q-Learning for Trajectory Balance

### Installation

- Create conda environment:

```sh
conda create -n gflownet-BM python=3.10
conda activate gflownet-BM
```

- Install PyTorch with CUDA. For our experiments we used the following versions:

```sh
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

or with pip see [most updated version](https://pytorch.org/get-started/locally/)

```sh
pip3 install torch torchvision torchaudio
```

You can change `pytorch-cuda=11.8` with `pytorch-cuda=XX.X` to match your version of `CUDA`.

- Install core dependencies:

```sh
pip install -r requirements.txt
```

-_(Optional)_ Install dependencies for molecule experiemtns

```sh
pip install -r requirements_mols.txt
```

### For MAC-OS USERS

Due to incompatibility of certain versions of `torch-sparse` and `torch-scatter`. You can run these commands

```sh
pip install git+https://github.com/rusty1s/pytorch_sparse.git
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html
pip --no-cache-dir install torch-geometric

```

### Hypergrids

Code for this part heavily utlizes library `torchgfn` (https://github.com/GFNOrg/torchgfn).

Path to configurations (utlizes `ml-collections` library):

- General configuration: `hypergrid/experiments/config/general.py`
- Algorithm: `hypergrid/experiments/config/algo.py`
- Environment: `hypergrid/experiments/config/hypergrid.py`

List of available algorithms:

- Baselines: `db`, `tb`, `subtb` from `torchgfn` library;
- Soft RL algorithms: `soft_dqn`, `munchausen_dqn`, `sac`.

Example of running the experiment on environment with `height=20`, `ndim=4` with `standard` rewards, seed `3` on the algorithm `soft_dqn`.

```bash
    python run_hypergrid_exp.py --general experiments/config/general.py:3 --env experiments/config/hypergrid.py:standard --algo experiments/config/algo.py:soft_dqn --env.height 20 --env.ndim 4
```

To activate learnable backward policy for this setting

```bash
    python run_hypergrid_exp.py --general experiments/config/general.py:3 --env experiments/config/hypergrid.py:standard --algo experiments/config/algo.py:soft_dqn --env.height 20 --env.ndim 4 --algo.tied True --algo.uniform_pb False
```

### Molecules Experiments

The presented experiments actively reuse the existing codebase for molecule generation experiments with GFlowNets (https://github.com/GFNOrg/gflownet/tree/subtb/mols).

You can change `requirements_mols.txt` to match your `CUDA` version by replacing `cu118` by `cuXXX`.

Additional requirements for molecule experiments:

- `pandas rdkit torch_geometric h5py ray hydra` (installation is available in `requirements_mols.txt`)

Path to configurations of `MunchausenDQN` (utilizes `hydra` library)

- General configuration: `mols/configs/soft_dqn.yaml`
- Algorithm: `mols/configs/algorithm/soft_dqn.yaml`
- Environment: `mols/configs/environment/block_mol.yaml`

To run `MunchausenDQN` with configurations prescribed above, use

```
    python soft_dqn.py
```

To reporoduce baselines, run `gflownet.py` with required parameters, we refer to the original repository https://github.com/GFNOrg/gflownet for additional details.

### Bit sequences Experiment

Examples of running `TB`, `DB` and `SubTB` baselines for word length `k=8`:

```
python bitseq/run.py --objective tb --k 8 --learning_rate 0.002
```

```
python bitseq/run.py --objective db --k 8 --learning_rate 0.002
```

```
python bitseq/run.py --objective subtb --k 8 --learning_rate 0.002 --subtb_lambda 1.9
```

Example of running `SoftDQN`:

```
python bitseq/run.py --objective softdqn --m_alpha 0.0 --k 8 --learning_rate 0.002 --leaf_coeff 2.0
```

Example of running `MunchausenDQN`:

```
python bitseq/run.py --objective softdqn --m_alpha 0.15 --k 8 --learning_rate 0.002 --leaf_coeff 2.0
```

## Local Search Experiment with adaptive Metropolis-Hastings algorithm

### Environment Setup

To install dependecies, please run the command `pip install -r requirement.txt`.
Note that python version should be < 3.8 for running RNA-Binding tasks. You should install `pyg` with the following command

```
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

### Code references

Our implementation is based on "Towards Understanding and Improving GFlowNet Training" (https://github.com/maxwshen/gflownet).

### Our contribution (in terms of codes)

We extend codebase with RNA-binding tasks designed by FLEXS (https://github.com/samsinai/FLEXS)

We implement detailed balance (DB), sub-trajectory balance (SubTB), and our method LS-GFN on top of DB, SubTB, TB, MaxEnt and GTB.

We also implement various state-of-the-art baselines including RL approaches (A2C-Entropy, SQL, PPO) and recent MCMC approaches (MARS)

### Large files

You can download additional large files by following link: [https://drive.google.com/drive/folders/1JobUWGowoiQxGWVz3pipdfcjhtdY4CVq?usp=sharing](https://drive.google.com/drive/folders/1zc9U5oWATEMps-FrmI_VPm7HBinZiKH-?usp=drive_link)

These files should be placed in `datasets`

### Molecule Experiments

```
# LS-GFN
conda info --envs
python testing/main.py --setting qm9str --model gfn --ls --seed 0

```

You can run the following command to validate the effectiveness of LS-GFN on various biochemical tasks.
As a default setting, we choose TB as a training objective and apply deterministic filtering to determine whether to accept or reject refined samples.

```
# LS-GFN
python main.py --setting qm9str --model gfn --ls --seed 0

# GFN (TB)
python main.py --setting qm9str --model gfn --seed 0
```

### Other Baselines

Beyond GFN baselines, we also implement reward-maximization methods as baselines. Baselines can be executed by setting `--model` option.

- Available Options: `mars, a2c, sql, ppo`

```
# MARS
python main.py --setting tfbind8 --model mars --seed 0

# A2C
python main.py --setting tfbind8 --model a2c --seed 0

# Soft Q-Learning
python main.py --setting tfbind8 --model sql --seed 0

# PPO
python main.py --setting tfbind8 --model ppo --seed 0
```

### Additional Experiments

You can change various biochemical tasks to evaluate the performance of LS-GFN by setting `--setting` option.

- Available Options: `qm9str, sehstr, tfbind8, rna1, rna2, rna3`

```
python main.py --setting <setting> --model gfn --ls --seed 0
```

You can change GFlowNet training objectives to evaluate the performance of Logit-GFN by setting `--loss_type` option.

- Available Options: `TB (Default), MaxEnt, DB, SubTB, GTB`

```
python main.py --setting qm9str --model gfn --ls --loss_type <loss_type> --seed 0
```

You can change the filtering strategies during local search by setting `--filtering` option.

- Available Options: `deterministic (Default), stochastic`

```
python main.py --setting qm9str --model gfn --ls --filtering <filtering_strategies> --seed 0
```

You can adjust the number of iterations per batch ($I$) by setting `num_iterations` option (Default: 7).

```
python main.py --setting qm9str --model gfn --ls --num_iterations <I> --seed 0
```

You can adjust the number of backtracking and reconstruction steps ($K$) by setting `num_back_forth_steps` option. (Default: $K=\lfloor(L+1)/2\rfloor$)

```
python main.py --setting qm9str --model gfn --ls --num_back_forth_steps <K> --seed 0
```

### Contribution

If you want to contribute you can fork this repo and create a new branch and add you own data

Just make sure to the following steps and create a PR.

```
git branch
git status
git branch -c gfn_model_branch
git checkout gfn_model_branch
git add .
git commit -m "add comment"
git push origin head
```
