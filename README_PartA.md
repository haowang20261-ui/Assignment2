# MLDS 490 — Homework 2, Part (a): CartPole-v1 with DQN

[Back to homework overview](README.md)

## 1. Objective

Implement a custom **Deep Q-Network (DQN)** agent for `CartPole-v1` and demonstrate learning through: (i) a maximum-Q-value curve, (ii) an episode-reward curve with a trailing 100-episode mean, and (iii) a reward histogram from **500 separate greedy evaluation episodes**, including the mean and standard deviation.

CartPole observations have **4 continuous features** and **2 discrete actions**. Episodes end at failure or the environment's **500-step** time limit. The discount factor is **0.95**. We use an online neural network, replay memory, and a separate target network.

## 2. Environment and installation

- **Experiment system:** Windows, Miniconda, Python 3.11; PyTorch training on available CPU/GPU.
- **Python packages:** `torch`, `gymnasium`, `numpy`, `pandas`, `matplotlib`, `jupyterlab`.

Install PyTorch using the [official CPU/CUDA instructions](https://pytorch.org/get-started/locally/) and then run:

```bash
python -m pip install jupyterlab gymnasium numpy pandas matplotlib
```

Start the notebook server from the folder containing the submitted notebook and `results_cartpole_dqn`:

```bash
jupyter lab
```

**Notebook identification:** The completed CartPole notebook was previously named `MLDS490_HW2_DQN_CartPole (1).ipynb`. Earlier development versions include `MLDS490_HW2_DQN_CartPole_Step01.ipynb` and `MLDS490_HW2_DQN_CartPole_Step02.ipynb`; **those early-stage notebooks do not contain the complete training experiment**. If the uploaded completed notebook has been renamed, open its actual filename rather than assuming an old development filename is the final submission.

## 3. Network and training configuration

| Setting | Value |
|---|---|
| Environment | `CartPole-v1` |
| Original seed | `42` |
| Discount factor | `gamma = 0.95` |
| Input / output dimensions | 4 / 2 |
| Q-network | Linear(4,128) → ReLU → Linear(128,128) → ReLU → Linear(128,2) |
| Learnable parameters | 17,410 |
| Replay capacity | 50,000 transitions |
| Replay warm-up | 1,000 environment steps |
| Mini-batch size | 64 |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| TD loss | Huber (smooth L1) |
| Gradient clipping | Maximum gradient norm 10 |
| Target network | Hard update every 1,000 environment steps |
| Exploration | Epsilon-greedy, `1.0` → `0.05` over 15,000 steps |
| Training budget | 100,000 environment steps |
| Training records | 1,018 episodes |
| Independent evaluation | 500 greedy episodes |

The network emits **raw Q-values** (no softmax). The maximum-Q statistic is the per-episode maximum over visited states and both available actions, as defined in the training notebook.

## 4. Algorithm pseudocode

```text
Initialize online Q-network Q_online and target network Q_target
Copy Q_online weights into Q_target
Initialize experience replay with capacity 50,000
Set Adam optimizer, learning rate 0.001, discount gamma = 0.95
Set environment interaction counter = 0

while interaction counter < 100,000:
    Reset CartPole and obtain initial observation
    repeat until episode ends or training budget is exhausted:
        Compute epsilon according to the linear exploration schedule
        With probability epsilon, select a random action
        Otherwise, select argmax_a Q_online(state, a)
        Step the environment
        Store (state, action, reward, next_state, terminated, truncated)

        After warm-up, sample replay mini-batches when scheduled:
            Evaluate Q_online(state, action)
            Construct TD target using target-network maximum Q-value
            Do not bootstrap from genuinely terminated states
            Compute Huber TD loss
            Backpropagate, clip gradient norm, and update Q_online

        Every 1,000 environment steps, copy Q_online to Q_target
        Record episode reward, maximum Q, and other diagnostics

    At validation intervals, test greedy policy on validation episodes
    Save a checkpoint when validation performance improves

Save the final model, histories, and figure data
Evaluate validation-best checkpoint on 500 independent episodes
```

**Termination:** `terminated` and `truncated` are distinct Gymnasium flags; a time limit should not be automatically treated as an environment failure when deciding whether to bootstrap TD targets.

## 5. How to execute

**Full training reproduction**

1. Start Jupyter in the repository root and open the **completed Part (a) notebook**.
2. Execute environment checks, network definition, replay memory, target-network and optimizer setup in notebook order.
3. Run the complete DQN training cells to the **100,000-step** budget. This is the expensive stage; do not rerun it merely to regenerate report images.
4. Run the plot/summary cells using saved `training_history.csv` and `validation_history.csv`.
5. Load `results_cartpole_dqn/dqn_best.pt` for **greedy** evaluation over 500 episodes. Do not confuse `dqn_best.pt` with `dqn_final.pt` or `dqn_latest.pt`.
6. Execute the notebook's edge-case, baseline, checkpoint, and behavioral-visualization tests as appropriate.

**Inspecting saved results without retraining**

From the repository root:

```bash
python -c "import pandas as pd; p='results_cartpole_dqn/training_history.csv'; d=pd.read_csv(p); print('Recorded episodes:',len(d)); print(d.tail(3).to_string(index=False))"
```

This command only reads saved output. Checkpoint evaluation needs the network class and evaluation cells/scripts included with the complete notebook; a `.pt` checkpoint alone is not an executable agent.

## 6. Recorded results and example output

The completed run produced the following observations:

| Metric | Recorded value |
|---|---:|
| Total training environment steps | 100,000 |
| Training episode records | 1,018 |
| Best validation checkpoint step | 55,198 |
| Best checkpoint validation mean | 500.0 |
| Final checkpoint validation mean | 97.45 |
| Independent greedy evaluation episodes | 500 |
| Independent greedy evaluation mean | **499.734** |
| Independent greedy evaluation sample standard deviation | **5.948** |
| Independent evaluation success rate (500-step episodes) | **99.8%** |
| Independent evaluation reward minimum / maximum | 367 / 500 |

Example summary (from recorded measurements, not a fresh execution):

```text
Training steps:                    100000
Best checkpoint step:               55198
500-episode evaluation mean:       499.734
500-episode sample std:              5.948
500-step success rate:              99.8%
```

### Validation and baseline comparisons

A separate **40-episode validation comparison** yielded:

| Policy | Mean reward |
|---|---:|
| Best checkpoint (greedy) | 500.0 |
| Final checkpoint (greedy) | 101.8 |
| Best checkpoint with epsilon = 0.25 | 48.025 |
| Random policy | 21.7 |

These results show that both checkpoint choice and exploration policy materially affect evaluation scores. The small 40-episode comparison is separate from the main 500-episode independent evaluation.

### Non-standard plot: action-preference map

An additional diagnostic visualizes the learned preference for left versus right across CartPole states. The recorded map covered **18,271 states**, with **64.83%** preferring right and **35.17%** left, and a **98.44%** mirrored-action agreement in the symmetry check. This is a behavioral diagnostic, not a substitute for independent returns.

## 7. Output files

Recorded artifact names include:

```text
results_cartpole_dqn/
├── training_history.csv
├── validation_history.csv
├── training_config.json
├── network_architecture.json
├── final_report_summary.json
├── dqn_best.pt                 # validation-best model
├── dqn_latest.pt               # resumable/latest state
├── dqn_final.pt                # final-step model
├── dqn_action_preference_grid.csv
├── dqn_action_preference_summary.json
├── dqn_action_preference_map.png
└── figures/                    # figures may also be copied to Output/ for Overleaf
```

Plots generated in the experiment and/or copied to the Overleaf `Output/` directory:

```text
Output/dqn_max_q.png
Output/dqn_training_rewards.png
Output/dqn_validation.png
Output/dqn_test_500_histogram.png
Output/dqn_action_preference_map.png
```

The experimental figure files were also exported in PDF form where recorded. Output locations may differ slightly between the notebook result directory and its Overleaf-ready copied figures.

## 8. Validation checklist

- Confirm `CartPole-v1` observation shape `(4,)`, 2 actions, and 500-step limit.
- Confirm online and target Q-networks give finite values of shape `(batch,2)`.
- Confirm replay sample and bootstrapping logic distinguish termination from time truncation.
- Confirm fixed-seed evaluation reproducibility and that NoisyNet is **not** used in Part (a).
- Compare greedy best, greedy final, epsilon-greedy, and random policies at matched evaluation conditions.
- Plot the three assignment figures and keep the numerical summaries with the submission.
- Run the non-standard action-preference map as an additional behavioral check.

## 9. Suggested three-month extensions

Repeat the training with multiple random seeds, ablate replay size and target-network update frequency, compare exploration schedules and learning rates, test observation perturbations, and analyze whether action-preference symmetry predicts generalization. Use environment-step-matched runs and separate validation and held-out test seeds.

## 10. Submission notes

Submit the **completed** Part (a) notebook, source code, README, algorithm pseudocode, and actual sample output; do not submit only early development notebooks. Ensure GitHub repository permissions and instructor access satisfy course requirements, and make sure the required Northwestern email is linked to the GitHub account.
