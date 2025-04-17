# train.py

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

from casino_env import CasinoEnv


cwd = Path().cwd()

def make_env():
    return CasinoEnv(
        num_tables=3,
        step_penalty=0.0,
        default_bet=1,
        illegal_penalty=-1.0,
    )

def get_train_val_env():
    train_env = SubprocVecEnv([make_env] * 32, start_method="fork")
    train_env = VecMonitor(train_env)

    eval_env = SubprocVecEnv([make_env] * 16, start_method="fork")
    eval_env = VecMonitor(eval_env)

    return train_env, eval_env

def get_dqn(train_env, device):
    return DQN(
        policy="MultiInputPolicy",
        env=train_env,
        device=device,
        tensorboard_log=f"{str(cwd)}/tensorboard/dqn/",
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
        verbose=1,
    )

def get_ppo(train_env, device):
    return PPO(
        policy="MultiInputPolicy",
        env=train_env,
        device=device,
        tensorboard_log=f"{str(cwd)}/tensorboard/ppo/",
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

def main(model_type: Literal["DQN", "PPO"]):
    models_path = cwd / "models"
    logs_path = cwd / "logs"
    models_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    train_env, eval_env = get_train_val_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{str(logs_path)}/{model_type.lower()}/",
        log_path=f"{str(logs_path)}/{model_type.lower()}/",
        eval_freq=10_000,
        n_eval_episodes=100,
        deterministic=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type == "DQN":
        model = get_dqn(train_env, device)
    elif model_type == "PPO":
        model = get_ppo(train_env, device)
    else:
        raise ValueError("Invalid model type. Choose 'DQN' or 'PPO'.")

    model.learn(
        total_timesteps=1_000_000,
        callback=eval_callback,
        log_interval=1000,
    )
    model.save(f"{str(models_path)}/{model_type.lower()}_blackjack")

    final_env = CasinoEnv(
        num_tables=3,
        step_penalty=0.0,
        default_bet=1,
        illegal_penalty=-1.0,
    )
    rewards = []
    for _ in range(100):
        obs = final_env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = final_env.step(action)
            total += r
        rewards.append(total)
    print(f"{model_type} average reward: {np.mean(rewards):.4f}")


if __name__ == "__main__":
    main(model_type="PPO")
