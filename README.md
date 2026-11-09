# MLDS 490 — Homework 2: Deep Q-Networks

**Repository:** [haowang20261-ui/Assignment2](https://github.com/haowang20261-ui/Assignment2)  
**Language:** Python 3.11 / Jupyter Notebook  
**Framework:** PyTorch and Gymnasium

This homework implements and evaluates two value-based reinforcement learning agents:

| Part | Environment | Method | Training budget | Independent evaluation |
|---|---|---|---|---|
| [Part (a)](README_PartA.md) | `CartPole-v1` | Custom Deep Q-Network (DQN) | 100,000 environment steps | 500 greedy episodes |
| [Part (b)](README_PartB.md) | `ALE/MsPacman-v5` | Modified Rainbow DQN | 5,000 episodes / 4,254,796 environment steps | 500 greedy episodes |

Both implementations include the **three required DQN components**: a neural network, an experience replay buffer, and a separate target network.

## Results at a glance

| Metric | Part (a): CartPole | Part (b): Ms. Pac-Man |
|---|---:|---:|
| Best checkpoint | 55,198 environment steps | Episode 2,900 |
| Mean reward (500 independent episodes) | **499.734** | **2,847.64** |
| Sample standard deviation | 5.948 | 984.67 |
| Maximum possible episode length / raw score note | 500 steps | Original, unclipped Atari game score |

**Important:** Results were obtained from *selected validation-best checkpoints*, not necessarily the final training checkpoints. Part (b) uses clipped rewards for training targets but reports **unclipped game scores**.

## Documentation

- **[Part (a) README](README_PartA.md):** architecture, hyperparameters, DQN pseudocode, execution and evaluation, outputs, edge cases, and non-standard plot.
- **[Part (b) README](README_PartB.md):** preprocessing, Rainbow components, training configuration and pseudocode, independent testing, baselines, reproducibility, and behavioral plot.

## Setup

Create or activate a Python 3.11 environment. Install PyTorch using the [official installation instructions](https://pytorch.org/get-started/locally/) appropriate for your CPU/CUDA setup. Then install the other packages:

```bash
python -m pip install jupyterlab numpy pandas matplotlib pillow "gymnasium[atari]"
```

Start Jupyter from the repository root:

```bash
jupyter lab
```

See the part-specific READMEs for the notebook sequence and how to reproduce saved figures and test statistics. Some analysis figures were deliberately generated from saved CSV/JSON data in a **separate Matplotlib/Agg Python subprocess** to avoid Windows OpenMP and Jupyter-kernel crashes.

## Suggested submission contents

- Completed Part (a) and Part (b) Jupyter notebooks, with source cells intact.
- Any auxiliary `.py` scripts referenced by those notebooks (especially independent evaluation and validation runners).
- This README and the two part-specific README files.
- Sample output: training summaries, evaluation CSV/JSON files, and the required plots.
- Model checkpoints needed to reproduce evaluation, if repository storage limits allow.
- PDF report with formulas, three required figures for each part, validation tests, non-standard plot, and three-month research proposals.

**Repository verification before submission:** ensure the repository is public *or* shared with the instructor as required; grant access to GitHub user `yoonsikHong` if the course requires a collaborator; verify the account has the required `northwestern.edu` email linked; and open the repository in a signed-out/private window to confirm source, README, and sample output are accessible. Large checkpoints and replay states may require Git LFS or another explicitly documented distribution method; do not silently omit the files required for evaluation.

## Notes on paths and reproducibility

The documented result paths are those used during the experiments. The final location of a full training notebook may depend on how the notebook is named when uploaded to GitHub. **Run the submitted complete notebook**, rather than the early `Step01`/`Step02` development notebooks alone; the latter validate individual components and do not perform the full training run.
