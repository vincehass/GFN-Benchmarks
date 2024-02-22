# Generative Flow Networks as Benchmark

## Installation

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

You can change `requirements_mols.txt` to match your `CUDA` version by replacing `cu118` by `cuXXX`.

## Hypergrids

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

## Molecules

The presented experiments actively reuse the existing codebase for molecule generation experiments with GFlowNets (https://github.com/GFNOrg/gflownet/tree/subtb/mols).

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

## Bit sequences

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
