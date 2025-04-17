# evaluate.py

from pathlib import Path
from typing import Literal

from gym import Wrapper
from casino_env import CasinoEnv
from stable_baselines3 import PPO, DQN


cwd = Path().cwd()

ACTION_NAMES = {
    0: "Join Table 0",
    1: "Join Table 1",
    2: "Join Table 2",
    3: "HIT",
    4: "STAND",
    5: "DOUBLE",
    6: "SPLIT"
}


class LoggingWrapper(Wrapper):
    """
    A thin wrapper that records every step, including full end‑of‑round context.
    It snapshots the player hand *before* calling env.step(..) so we still have
    access even if the environment clears hands when `done=True`.
    """

    def __init__(self, env):
        super().__init__(env)
        self.log = []
        self.cur_cards = []
        self.dealer_final = None
        self.player_final = None

    # ─────────────────────────── helpers ────────────────────────────
    def _flatten_obs(self, obs):
        return {
            "phase":          obs["phase"],
            "balance":        round(float(obs["balance"][0]), 4),
            "true_count":     round(float(obs["true_count"][0]), 3),
            "in_hand_total":  round(obs["in_hand"][0] * 21, 1),
            "is_soft":        bool(obs["in_hand"][1]),
            "dealer_upcard":  round(obs["in_hand"][2] * 11, 1),
            "can_double":     bool(obs["in_hand"][3]),
            "has_pair":       bool(obs["in_hand"][4]),
            "num_cards":      round(obs["in_hand"][5] * 11, 1),
        }

    # ───────────────────────── episode control ──────────────────────
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.log = [{
            "step": 0, "phase": obs["phase"], "obs": self._flatten_obs(obs),
            "action": None, "action_str": None, "reward": 0.0,
            "done": False, "info": "reset"
        }]
        self.cur_cards, self.dealer_final, self.player_final = [], None, None
        return obs

    def step(self, action):
        # ░░ snapshot hand BEFORE env.step() (env may clear it on `done`) ░░
        pre_hand = None
        if self.env.phase == 1 and self.env.agent.hands:
            pre_hand = self.env.agent.hands[self.env.hand_index]

        # ░░ take the actual environment step ░░
        obs, reward, done, info = self.env.step(action)

        # ░░ during play (phase==1) keep current cards for nice logging ░░
        if self.env.phase == 1 and self.env.agent.hands:
            self.cur_cards = [str(c) for c in self.env.agent.hands[self.env.hand_index].cards]

        # ░░ end‑of‑round capture ░░
        if done:
            # dealer
            if self.env.current_table and self.env.current_table.dealer.hand:
                h = self.env.current_table.dealer.hand
                self.dealer_final = {"hand": [str(c) for c in h.cards], "value": h.value_hard}

            # player (use pre‑snapshot because env may have wiped hands)
            if pre_hand:
                self.player_final = {
                    "hand": [str(c) for c in pre_hand.cards],
                    "value": pre_hand.value_hard
                }
                self.cur_cards = self.player_final["hand"]

        # ░░ log entry ░░
        self.log.append({
            "step":  len(self.log),
            "phase": obs["phase"],
            "obs":   self._flatten_obs(obs),
            "action": action,
            "action_str": ACTION_NAMES.get(
                action.item() if hasattr(action, "item") else action,
                str(action)
            ),
            "reward": reward,
            "done":   done,
            "dealer_final": self.dealer_final if done else None,
            "player_final": self.player_final if done else None,
            "cards": self.cur_cards.copy(),
        })
        return obs, reward, done, info

    # ─────────────────────────── pretty print ───────────────────────
    def summarize_log(self):
        print("=" * 60)
        for e in self.log:
            if e["action"] is None:   # reset line
                continue

            p   = e["phase"]
            obs = e["obs"]
            act = f"{e['action_str']:12}"
            illegal = "Illegal" if e["reward"] < -0.5 and p == 1 else ""
            cards   = f" | Cards: {e['cards']}" if e["cards"] else ""

            if e["done"]:
                d_val = e["dealer_final"]["value"]
                p_val = e["player_final"]["value"]
                busted = "💥 BUSTED" if p_val > 21 else ""
                print(f"Step {e['step']}: {act} → END")
                print(f"Dealer: {d_val}  |  Player: {p_val} {busted}  "
                      f"→ reward {e['reward']}")
                print("-" * 60)
            elif p == 0:
                # sitting at a table (no in‑hand metrics yet)
                print(f"Step {e['step']}: {act}")
            else:
                # in‑hand decision
                print(f"Step {e['step']}: {act} "
                      f"→ {int(obs['in_hand_total']):2} vs {int(obs['dealer_upcard']):2} "
                      f"{illegal}{cards}")


def main(model_type: Literal["DQN", "PPO"]):
    model_path = cwd / "models" / f"{model_type.lower()}_blackjack.zip"
    if model_type == "PPO":
        model = PPO.load(model_path)
    elif model_type == "DQN":
        model = DQN.load(model_path)
    else:
        raise ValueError("Invalid model type. Choose 'DQN' or 'PPO'.")

    env = LoggingWrapper(CasinoEnv(num_tables=3))
    for _ in range(100):
        done = False
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

        env.summarize_log()


if __name__ == "__main__":
    main(model_type="DQN")
