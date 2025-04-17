# benchmark.py

from pathlib import Path
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from collections import Counter
from stable_baselines3 import PPO, DQN

from casino_env import CasinoEnv
from game import Settings


cwd = Path().cwd()

def play_hand(model_type: str, deterministic: bool = True):
    env = CasinoEnv(
        num_tables=3,
        step_penalty=0.0,
        default_bet=1,
        illegal_penalty=-1.0
    )
    model_path = cwd / "models" / f"{model_type.lower()}_blackjack.zip"
    if model_type == "PPO":
        model = PPO.load(model_path, device="cpu")
    elif model_type == "DQN":
        model = DQN.load(model_path, device="cpu")
    else:
        raise ValueError("Invalid model type. Choose 'DQN' or 'PPO'.")

    obs = env.reset()
    total_reward = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

    outcome = 1 if total_reward > 0 else 0 if total_reward == 0 else -1
    bj = abs(total_reward - Settings.blackjack_payout) < 1e-6
    return total_reward, outcome, bj

def evaluate_fast(model_type: str, n_hands=100_000, n_jobs=16):
    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(play_hand)(model_type) for _ in range(n_hands)
    )

    rewards, outcomes, bjs = zip(*results)
    rewards = np.array(rewards)

    out_counts = Counter(outcomes)
    streaks = []
    current = 0
    for o in outcomes:
        if o == 1:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)

    return {
        "win_rate": out_counts[1] / n_hands,
        "push_rate": out_counts[0] / n_hands,
        "loss_rate": out_counts[-1] / n_hands,
        "avg_reward": np.mean(rewards),
        "bj_rate": sum(bjs) / n_hands,
        "streak_dist": dict(Counter(streaks)),
    }

def main(model_type: Literal["DQN", "PPO"]):
    stats = evaluate_fast(model_type, n_hands=100_000, n_jobs=16)

    print(f"Model: {model_type}")
    print(f"Win rate: {stats['win_rate']:.3%}")
    print(f"Push rate: {stats['push_rate']:.3%}")
    print(f"Loss rate: {stats['loss_rate']:.3%}")
    print(f"BJ rate: {stats['bj_rate']:.3%}")
    print(f"Avg reward: {stats['avg_reward']:.4f}")
    print("Win‐streak distribution (length: count):")
    for length, cnt in sorted(stats["streak_dist"].items()):
        print(f"{length:2d}: {cnt}")


if __name__=="__main__":
    main(model_type="PPO")
