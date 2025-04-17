[# 🂡 Blackjack Reinforcement‑Learning Agent

> **Beat the dealer — or at least try.**  
> A fully‑custom Blackjack engine wrapped in a Gym‑style environment and trained with
> modern RL algorithms (PPO & DQN) using [Stable‑Baselines 3](https://github.com/DLR-RM/stable-baselines3).
> The environment allows agents to learn blackjack strategies with optional card-counting signals and evaluates their performance comprehensively.

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) ![python](https://img.shields.io/badge/python-3.9%2B-blue.svg)

---

## Table of Contents
1. [Motivation](#motivation)
2. [Project Layout](#project-layout)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Environment Design](#environment-design)
6. [Training Scripts](#training-scripts)
7. [Benchmark Results](#benchmark-results)
8. [Interpreting the Results](#interpreting-the-results)
9. [How It Works](#how-it-works)
10. [Future Work](#future-work)
11. [License & Attribution](#license--attribution)

---

## Motivation
Casino blackjack serves as an excellent benchmark for sequential decision‑making under uncertainty. This project provides a rules‑accurate, multi‑deck blackjack simulator integrated with Gym and explores two canonical deep RL algorithms:

- **Deep Q-Networks (DQN)**
- **Proximal Policy Optimization (PPO)**

Agents are trained to:

- Select optimal tables using normalized *true count* signals
- Decide actions (**HIT / STAND / DOUBLE / SPLIT**)
- Manage bankroll effectively over extended sessions

The environment emphasizes **partial observability**, **delayed rewards**, and **illegal-action masking** for robust RL training.

This project implements a **two‑stage Markov Decision Process** for blackjack:

1. **Table‑selection phase** – the agent chooses which table (shoe) to join, implicitly
   learning card‑counting via the shoe’s _true count_ feature.
2. **In‑hand phase** – classic blackjack actions: **hit · stand · double · split**.

A handcrafted game engine (`game.py`) guarantees determinism and separates game
logic from RL concerns.  The Gym wrapper (`casino_env.py`) exposes a **multi‑input
observation space** combining balance, card counts, dealer up‑card and an
action‑mask for legality‑aware exploration.

Training, evaluation and large‑scale Monte‑Carlo benchmarking scripts are
included so you can reproduce every number in this README with a single GPU.

---

## Project Layout
```
.
├── game.py            # Core blackjack engine (cards, shoe, dealer logic)
├── casino_env.py      # Gym environment with observation/action spaces, masking, and rewards
├── train.py           # Training pipeline for PPO & DQN, TensorBoard logging
├── evaluate.py        # Detailed single‑episode evaluation
├── benchmark.py       # Parallel Monte‑Carlo evaluation (100 k hands)
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Installation
```bash
git clone https://github.com/amirmasoudaz/blackjack-rl.git
cd blackjack-rl
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**Dependencies:**
* `stable‑baselines3>=2.3`
* `gymnasium` (Gym compatibility layer shipped with SB3)
* `torch` (GPU if available)
* `tqdm`, `joblib`, `numpy`

---

## Quick Start
### Training agents
```bash
# PPO (default 1 M timesteps ≈ 20 min on RTX 4060)
python train.py --model PPO
# DQN
python train.py --model DQN
```
TensorBoard logs are written to `tensorboard/{ppo|dqn}`.

### Evaluate trained agents
```bash
python evaluate.py --model_type PPO  # or DQN
```

### Run Monte-Carlo benchmark (100,000 hands, takes ~3min on 32 CPU cores)
```bash
python benchmark.py --model_type PPO    # or DQN
```

---

## Environment Design

| Element              | Description                                                                                      |
|----------------------|--------------------------------------------------------------------------------------------------|
| **Decks & Shuffle**  | 8 decks, reshuffle after 45–55 % of shoe played                                                  |
| **Rules**            | Dealer hits soft 17, no surrender or splits, DOUBLE allowed                                      |
| **Betting**          | Fixed bet with optional DOUBLE                                                                   |
| **Counting**         | Hi-Lo true count as normalized input                                                             |
| **Observation**      | `Dict(phase, balance, table_info, in_hand, true_count, action_mask)`                             |
| **Reward Structure** | +1 (win), 0 (push), -1 (loss), +1.5 (blackjack), step/illegal-action penalties                   |
| **Actions**          | Table selection, HIT, STAND, DOUBLE, SPLIT (masked for legality)                                 |

---

## Training Scripts

- **`train.py`**
  - Parallel environments (`SubprocVecEnv`, 32 copies)
  - TensorBoard integration for monitoring and logging
  - CUDA automatic detection and optimization

- **Agent configurations** ensure comparable training conditions for PPO and DQN, highlighting algorithmic differences clearly.

---

## Benchmark Results

| Metric (100 k hands) |  **DQN**   |   **PPO**   |
|----------------------|:----------:|:-----------:|
| Win rate             |   40.96%   | **42.02%**  |
| Push rate            | **9.62%**  |    7.69%    |
| Loss rate            | **49.42%** |   50.30%    |
| Blackjack rate       | **4.52%**  |    4.49%    |
| Avg reward / hand    |   −0.070   | **−0.060**  |

---

## Interpreting the Results
- Both agents maintain negative expected values due to inherent casino odds (~0.6% house edge).
- PPO slightly outperforms DQN in terms of win rate (by <2%) and average returns (1 cents per dollar), highlighting PPO's efficiency. Hyper‑parameter tuning or Rainbow‑style improvements may close the gap.
- Illegal-action masking significantly improves learning stability and reduces unnecessary exploration.
- Step‑wise illegal‑action penalties speed up learning stability in DQN.
- Future enhancements in bet sizing and card-counting could further improve agent profitability.

---

## How It Works
### Environment Design
* **Observation Space** (multi‑input dict)
  * *phase* ∈ {0,1}
  * *balance* ∈ [0,1]
  * *table_info* (shape =`(N_tables,4)`) – min bet, always‑1 placeholder, clipped
    true‑count, occupancy flag
  * *in_hand* (shape =`(6,)`) – player total, softness flag, dealer up‑card, can‑double,
    pair flag, card count
  * *true_count* – scalar shoe count reused by counting agents
  * *action_mask* – dynamic legality mask concatenated to the observation

* **Reward Shaping**
  * Win = +1, Push = 0, Loss = −1, Blackjack = +1.5
  * Small step‑penalty (0) keeps Q‑values centered while allowing for additional
    shaping experiments.

### Training Details
* **PPO** – 256‑step rollouts, 10 epochs, `clip=0.2`, `ent_coef=0.01`, γ=0.99
* **DQN** – 100k replay buffer, `train_freq=1`, ε‑greedy decay 0.5→0.05, target
  network update every 10k steps
* **Vectorised Envs** – 32 parallel workers speed up sample collection ×30.

### Logging & Analysis
* **TensorBoard** – automatic scalar & histogram logging per algorithm
* **LoggingWrapper** – prints every decision with card details for debugging
* **Benchmark** – embarrassingly parallel Monte‑Carlo evaluator using `joblib`

---

## Future Work
- [ ] Multi‑agent training (cooperative vs. competitive)
- [ ] Implement a more complex reward structure (e.g., risk-adjusted returns)
- [ ] Enable splits, insurance, and multi-hand strategies
- [ ] Implement advanced RL algorithms (Rainbow DQN and Distributional RL)
- [ ] Introduce variable betting and bankroll management
- [ ] Curriculum learning based on deck penetration and complexity

---

## License & Attribution

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

If you find this project useful or build something on top of it, feel free to mention or link back to this repo — it'd mean a lot!

---]()