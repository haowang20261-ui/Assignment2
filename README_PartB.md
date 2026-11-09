# MLDS 490 — Homework 2, Part (b): Modified Rainbow DQN for Ms. Pac-Man

[Back to homework overview](README.md)

## 1. Objective

Train a **Modified Rainbow Deep Q-Network** on Atari `ALE/MsPacman-v5`, and document convergence and game performance with: (i) predicted Q-values versus training episodes, (ii) raw episode rewards with a 100-episode moving average, and (iii) the raw-reward histogram, mean and standard deviation of **500 independent evaluation episodes**. Include meaningful edge cases, performance validation, one **non-standard** plot, and a proposed three-month validation plan.

The agent satisfies all mandatory DQN components: a neural-network value function, replay buffer, and target network. Rainbow additionally combines distributional Q-learning (C51), Double DQN, dueling value/advantage heads, NoisyNet exploration, prioritized replay, and multi-step returns.

## 2. Environment and installation

| Property | Value |
|---|---|
| Gymnasium ID | `ALE/MsPacman-v5` |
| ALE version observed | `0.12.1` |
| Raw observation | RGB `(210,160,3)`, `uint8` |
| Action space | 9 discrete actions |
| Preprocessed single frame | `(88,80,1)`, signed `int8` |
| Network state | Four stacked frames, `(4,88,80)` |
| Discount factor | `0.99` |
| Default Atari v5 behavior used | Frame skip 4, sticky-action probability 0.25 |
| Device used | NVIDIA GeForce RTX 4090 Laptop GPU, CUDA-enabled PyTorch |
| Python | 3.11.15 on Windows / Miniconda |

Install PyTorch for the local CPU/CUDA configuration using the [official installation page](https://pytorch.org/get-started/locally/). Install the remaining packages:

```bash
python -m pip install jupyterlab "gymnasium[atari]" numpy pandas matplotlib pillow
```

If the environment cannot find Atari ROMs, follow the documentation for the installed `ale-py` version. Do **not** add a second frame-skipping wrapper: the default v5 environment already applies frame skip.

Start Jupyter in the repository root:

```bash
jupyter lab
```

Previously used development notebooks include `MLDS490_HW2_PartB_Step1.ipynb` (Atari/preprocessing checks) and `MLDS490_HW2_PartB_Step2.ipynb` (network checks). These are **component tests, not the complete 5,000-episode training notebook**. Run the full training notebook included with the submitted repository, following its cell order. The precise filename of that completed notebook should be checked against the uploaded repository before submission.

## 3. Preprocessing

The assignment-specified preprocessing:

1. Crop rows `1:176:2` and downsample columns `::2`.
2. Sum the RGB values to one intensity channel.
3. Set pixels matching the defined Ms. Pac-Man color (`210+164+74`) to zero.
4. Apply integer scaling `img // 3 - 128`, convert to signed `int8`, and reshape to `(88,80,1)`.
5. Stack four successive frames in channel-first order to obtain `(4,88,80)`.
6. Inside the network, normalize to floating point using `(state.float() + 128) / 255`.

**Reward distinction:** Training targets use **sign-clipped rewards** in `{-1,0,+1}`; training performance and evaluation use the original, **unclipped** Ms. Pac-Man game score.

## 4. Model and hyperparameters

| Setting | Value |
|---|---|
| Algorithm | Modified Rainbow DQN |
| Network | CNN encoder → 512-unit dense layer → dueling NoisyLinear streams |
| Convolutional layers | 4→32 (8×8, stride 4); 32→64 (4×4, stride 2); 64→64 (3×3, stride 1) |
| Flattened convolutional features | 2,688 |
| Learnable parameters | ~3,028,636 |
| Actions | 9 |
| C51 categorical support | 51 atoms spanning `[-10,10]` |
| Discount factor | `gamma = 0.99` |
| Multi-step return | 3 steps |
| Optimizer | Adam, learning rate `1e-4` |
| Replay buffer | Frame-efficient prioritized replay; 100,000-frame capacity |
| Prioritization exponent | `alpha = 0.6` |
| Importance sampling | `beta = 0.4` linearly increasing to 1.0 |
| Replay warm-up | 20,000 environment steps |
| Mini-batch size | 32 |
| Optimization cadence | Every 4 environment steps |
| Target-network synchronization | Every 8,000 environment steps |
| Training | 5,000 episodes / 4,254,796 environment steps |
| Optimization updates recorded | 1,058,700 |
| Independent test | 500 greedy episodes, seeds 900000–900499 |

The C51 distribution for each `(state, action)` is projected onto the fixed 51-point support. Greedy actions are selected by the **expected Q-value** of the distribution, not the largest categorical atom probability. Dueling value and advantage estimates are combined using mean-centered advantages. NoisyNet exploration is active during training and disabled during deterministic evaluation.

## 5. Algorithm pseudocode

```text
Initialize online and target Rainbow distributional networks
Copy online weights into target network
Initialize frame-based prioritized replay and 3-step return storage
Configure C51 support z_1...z_51 in [-10, 10]
Initialize Adam(learning_rate = 1e-4)

for episode in 1...5000:
    Reset ALE/MsPacman-v5 and initialize four-frame observation stack
    while not terminated/truncated:
        Obtain expected Q_online(state, each action) from categorical distributions
        Select greedy action under current training-time NoisyNet parameters
        Step environment and collect raw game reward
        Sign-clip reward for the learning target
        Append transition and form eligible 3-step experiences in replay
        Update observation frame stack and raw episode score

        Once warm-up is complete, every 4 environment steps:
            Sample prioritized replay batch with importance-sampling weights
            Choose next actions using online-network expected Q-values
            Obtain target distributions from target network (Double DQN)
            Build clipped, discounted 3-step distributional targets
            Project those targets back onto the 51-atom C51 support
            Compute importance-weighted cross-entropy loss
            Take Adam update and refresh replay priorities

        Every 8,000 environment steps, synchronize the target network

    Log raw episode score, mean loss, mean of maximum predicted Q-values
    At validation intervals, evaluate and save the highest-scoring checkpoint

Save final trained checkpoint and numerical histories
Run independent 500-episode greedy evaluation with exploration noise off
Run edge cases, random baseline, best-versus-final comparison, and plots
```

**Q-value curve definition:** The recorded training metric `mean_max_q` is a **mean of state-wise maximum expected Q-values over sampled states**. It is *not* the absolute maximum over all decisions in a training episode. The first 40 episodes had no logged Q-value mean during replay warm-up; report that distinction accurately.

## 6. How to run or reproduce

### A. Full training (computationally expensive)

1. Open and execute the complete Part (b) notebook in Jupyter from the repository root.
2. Check `ALE/MsPacman-v5`, nine actions, teacher preprocessing, and the `(4,88,80)` input stack.
3. Validate the C51 probability rows, NoisyNet behavior, forward shapes, gradients, prioritized replay, n-step data, and target-network logic.
4. Run or resume full Rainbow training up to **5,000 episodes**. Save checkpoints and CSV history.
5. Generate required training curves from saved CSV logs, not by rerunning training.
6. Load `rainbow_best.pt` for the independent **500-episode greedy** evaluation. Do not select the checkpoint on held-out test scores.
7. Execute the existing independent evaluation, edge-case, baseline, checkpoint-comparison, and behavioral visualization cells/scripts in the order documented in the notebook.

### B. Recheck the saved 5,000-episode training log without training

```bash
python -c "import pandas as pd; p='results_mspacman_rainbow/full_training/rainbow_training_episodes.csv'; d=pd.read_csv(p); print('Episodes:',len(d)); print('Environment steps:',int(d.environment_steps.iloc[-1])); print('First 100 mean:',d.raw_score.iloc[:100].mean()); print('Last 100 mean:',d.raw_score.iloc[-100:].mean())"
```

Expected values from the completed run:

```text
Episodes: 5000
Environment steps: 4254796
First 100 mean: 322.8
Last 100 mean: 2370.5
```

### C. Independent evaluation and validation

The evaluator `results_mspacman_rainbow/full_training/step8_independent_runner.py` defines a **standalone evaluation network and preprocessing**, and loads `rainbow_best.pt`. Use the notebook's existing evaluator-invoking cells, which set checkpoint paths, seeds, and subprocess arguments. **Do not assume that running the script with no command-line arguments reproduces the reported test.**

The edge-case runner `step9_edge_cases_runner.py` tests numerical properties and fixed-seed rollouts. The Step 13 behavioral plot uses `step13_action_map_data.json` as an intermediate file. In our Windows environment, collecting model/game trajectories and rendering Matplotlib figures were done in **separate subprocesses**, using `matplotlib`'s `Agg` backend, to prevent `libiomp5md.dll` duplicate OpenMP-runtime crashes. Replot saved JSON/CSV instead of rerunning game evaluation.

## 7. Results and example output

### Training and selected checkpoint

| Metric | Recorded result |
|---|---:|
| Training episodes | 5,000 |
| Environment steps | 4,254,796 |
| Mean raw score, first 100 training episodes | 322.8 |
| Mean raw score, last 100 training episodes | 2,370.5 |
| Maximum single training-episode score | 7,940 |
| Best validation checkpoint | Episode 2,900 |
| Best checkpoint validation mean score | 2,894.0 |

### Independent 500-episode greedy evaluation

| Metric | Recorded result |
|---|---:|
| Evaluation seeds | 900000–900499 |
| Mean raw score | **2,847.64** |
| Sample standard deviation | **984.67** |
| Median | 2,705 |
| Minimum / maximum | 1,060 / 7,490 |
| Approximate 95% confidence interval for mean | [2,761.33, 2,933.95] |

### Policy comparisons (same 500 test seeds)

| Policy | Mean raw score | Sample std. dev. |
|---|---:|---:|
| Random action policy | 252.98 | 104.47 |
| **Best Rainbow, episode 2900** | **2,847.64** | 984.67 |
| Final Rainbow, episode 5000 | 1,988.38 | 613.64 |

Best Rainbow beat the random policy on **500/500** paired seeds (11.26× mean score). Against the final checkpoint, best won **400**, final won **99**, and **1** tied; mean paired best-minus-final reward was **859.26** with approximate 95% confidence interval **[756.67, 961.85]**. This is evidence of late-stage policy degradation in this training run, not proof of the cause.

### Edge cases

All five tested categories passed: extreme int8 observation values; normalized 51-atom distributions and expected-Q support bounds; identical repeated rollout with fixed seed 910011 (score **5050**, **873** steps, same action digest); correct externally imposed 20-step evaluation cutoff; and nine-action diversity with separate raw versus clipped reward accounting (raw **5050**, clipped **138** for the diagnostic trajectory).

### Non-standard behavior visualization

The Action-Preference and Decision-Margin Map was made from **three held-out rollouts** at seeds **910100, 910101, 910102**. Each used all nine actions. Scores were **2620**, **3830**, **1760**; action switch rates **0.3732**, **0.3860**, **0.3128**; mean Q-value margins **0.0254**, **0.0230**, **0.0332**. The map shows selected action per decision step beside the gap between the largest and second-largest predicted Q-values. A large margin is **not** a calibrated probability of correctness.

## 8. Recorded files and sample output

The completed experiment generated files under:

```text
results_mspacman_rainbow/
├── step1_preprocessing_preview.png
├── step2_rainbow_network_validation.json
└── full_training/
    ├── rainbow_training_episodes.csv
    ├── rainbow_validation.csv
    ├── rainbow_best.pt
    ├── rainbow_final.pt
    ├── step8_independent_runner.py
    ├── step8_best_independent_scores.csv
    ├── step8_evaluation_summary.json
    ├── step9_edge_cases_runner.py
    ├── step9_edge_cases.json
    ├── step10_random_policy_scores.csv
    ├── step10_random_comparison.json
    ├── step11_final_independent_scores.csv
    ├── step11_best_vs_final_summary.json
    ├── step13_action_map_data.json
    └── ... additional training/resume and plotting artifacts
```

PNG figures previously generated in the report's `Output/` directory include:

```text
Output/mspacman_mean_max_q_curve.png
Output/mspacman_training_rewards.png
Output/mspacman_independent_reward_histogram.png
Output/mspacman_policy_comparison.png
Output/mspacman_best_final_paired_difference.png
Output/mspacman_action_preference_map.png
```

When uploading to GitHub, retain the original script and result filenames. Some figure copies may live in both the training folder and an `Output/` folder used by Overleaf. Ensure any figure paths in the submitted PDF match the files actually uploaded.

## 9. Validation experiments and proposed three-month extension

Beyond shape/unit checks, the completed work compared the best trained agent against a random policy and the final trained checkpoint using **matched independent seeds**, summarized uncertainty using confidence intervals, and examined decision sequences with a custom visualization.

With three additional months, prioritize: (1) **multi-seed training** to test reproducibility of late-stage degradation; (2) controlled **Rainbow component ablations** (PER, NoisyNet, 3-step, dueling); (3) learning-rate and target-sync schedules; (4) replay capacity and priority distributions; (5) robustness to sticky-action and observation perturbations; and (6) expanded action-margin/behavior analysis. Compare performance at matched environment-step budgets with a separate test set.

## 10. Submission notes

Include the completed training notebook and any auxiliary `.py` scripts required to evaluate checkpoints. Do not submit only preprocessing/network-validation notebooks. Include the README, source code, pseudocode, sample numerical output, report figures, and the saved best model when allowed by repository size. Verify public repository or instructor collaborator access, add `yoonsikHong` when required, and confirm the required `northwestern.edu` email is linked to GitHub.
