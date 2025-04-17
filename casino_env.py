# casino_env.py

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from game import Table, Player, Hand, Game, Logic, Settings


class CasinoEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, num_tables=5, step_penalty=0.0,
                 default_bet=1, illegal_penalty=-1.0,
                 bankruptcy_penalty=-10.0):
        super(CasinoEnv, self).__init__()
        self.np_random = None
        self.seed()

        self.num_tables = num_tables
        self.step_penalty = step_penalty
        self.default_bet = default_bet
        self.illegal_penalty = illegal_penalty
        self.bankruptcy_penalty = bankruptcy_penalty

        self.initial_balance = 100
        self.tables = [Table() for _ in range(self.num_tables)]
        self.agent = Player(name="Agent", balance=self.initial_balance)
        self.current_table = None
        self.phase = 0
        self.done = False
        self.game = None
        self.hand_index = 0

        self.observation_space = spaces.Dict({
            "phase": spaces.Discrete(2),
            "balance": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "table_info": spaces.Box(0.0, 1.0, shape=(self.num_tables, 4), dtype=np.float32),
            "in_hand": spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32),
            "true_count": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            "action_mask": spaces.Box(0.0, 1.0, shape=(self.num_tables + 4,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(self.num_tables + 4)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        import random
        random.seed(seed)
        return [seed]

    def reset(self):
        self.done = False
        self.phase = 0
        self.hand_index = 0
        self.agent.balance = self.initial_balance
        self.agent.clear_hands()
        for table in self.tables:
            if self.agent.table_pos is not None:
                table.remove_player(self.agent)
            table.shoe.build()
        self.current_table = None
        return self._get_observation()

    def step(self, action):
        info = {}

        obs = self._get_observation()
        action_mask = obs['action_mask']
        if action_mask[action] == 0:
            reward = self.illegal_penalty
            return obs, reward, self.done, {"illegal": True}

        reward = 0.0

        if self.phase == 0:
            if self.agent.balance < self.default_bet:
                self.done = True
                reward = self.bankruptcy_penalty
                info["bankrupt"] = True
                return self._get_observation(), reward, self.done, info
            if action < 0 or action >= self.num_tables:
                reward += self.illegal_penalty
                return self._get_observation(), reward, self.done, info

            table = self.tables[action]
            self.current_table = table
            if self.current_table.shoe.reshuffle_required:
                self.current_table.shoe.build()
            table.seat_player(self.agent, 0)
            table.dealer.start_new_hand()
            self.agent.clear_hands()
            self.agent.add_hand()
            self.agent.place_bet(self.agent.hands[0], self.default_bet)
            self.game = Game(table)
            self.game.deal_initial_cards()
            self.phase = 1
            return self._get_observation(), reward, self.done, info

        reward += self.step_penalty
        hands = self.agent.hands
        current = hands[self.hand_index]
        can_double = current.can_double and self.agent.balance >= current.bet
        legal = [current.can_hit, True, can_double, current.can_split]
        idx = action - self.num_tables
        if idx < 0 or idx >= 4 or not legal[idx]:
            reward += self.illegal_penalty
            return self._get_observation(), reward, self.done, {"illegal": True}
        else:
            # 0=hit,1=stand,2=double,3=split
            if idx == 0:
                card = self.current_table.shoe.deal()
                current.add_card(card)
            elif idx == 1:
                pass
            elif idx == 2:
                self.agent.place_bet(current, current.bet)
                card = self.current_table.shoe.deal()
                current.add_card(card)
                if current.value_hard >= 21:
                    reward -= 1.0
                if current.value_hard == 21:
                    reward += 1.0
            else:
                card1, card2 = current.cards
                current.cards = [card1]
                new_hand = Hand(role="player")
                new_hand.cards = [card2]
                new_hand.bet = current.bet
                self.agent.balance -= current.bet
                hands.insert(self.hand_index + 1, new_hand)
                current.add_card(self.current_table.shoe.deal())
                new_hand.add_card(self.current_table.shoe.deal())

        if idx in [1, 2] or (idx == 0 and current.has_busted) or (idx == 3 and not current.can_split) or not (
                0 <= idx < 4 and legal[idx]):
            self.hand_index += 1

        if self.hand_index >= len(hands):
            table = self.current_table
            table.dealer.dealer_logic(table.shoe)
            dealer = table.dealer.hand
            for hand in hands:
                # payout (push = bet, win = 2 * bet, BJ = 2.5 * bet)
                payout = Logic.judge_hands(hand, dealer)
                if payout > 0:
                    self.agent.collect_winnings(payout)
                # reward (bust = −1, push = 0, win = +1, BJ = +1.5)
                reward += Logic.judge_hands_norm(hand, dealer)
            self.done = True
            self.phase = 0
            self.hand_index = 0
            table.remove_player(self.agent)
            self.agent.clear_hands()

        return self._get_observation(), reward, self.done, info

    def _get_observation(self):
        bal = np.array([self.agent.balance / self.initial_balance], dtype=np.float32)
        phase = self.phase
        table_info = np.zeros((self.num_tables, 4), dtype=np.float32)
        tc_vals = 0.0
        true_count = np.array([0.0], dtype=np.float32)
        in_hand = np.zeros(6, dtype=np.float32)
        action_mask = np.zeros(self.num_tables + 4, dtype=np.float32)

        if self.phase == 0:
            for i, table in enumerate(self.tables):
                occ = int(any(p is not None for p in table.players.values()))
                tc = table.shoe.true_count() / Settings.number_of_decks
                table_info[i] = [
                    Settings.minimum_bet / Settings.maximum_bet,
                    1.0,
                    np.tanh(tc),
                    occ
                ]
            for i in range(self.num_tables):
                action_mask[i] = 1.0
        else:
            tc = self.current_table.shoe.true_count()
            tc_vals = np.tanh(tc)
            true_count = np.array([tc_vals], dtype=np.float32)
            hand = self.agent.hands[self.hand_index]
            total = hand.value_hard / 21.0
            is_soft = 1.0 if hand.value_soft > hand.value_hard else 0.0
            up = Settings.dealer_rank_value.get(self.current_table.dealer.upcard.rank, 10) / 11.0
            can_double = 1.0 if hand.can_double and self.agent.balance >= hand.bet else 0.0
            has_pair = 1.0 if hand.has_pair else 0.0
            num_c = len(hand.cards) / 11.0
            in_hand = np.array([total, is_soft, up, can_double, has_pair, num_c], dtype=np.float32)
            legal = [hand.can_hit, True, hand.can_double and self.agent.balance >= hand.bet, hand.can_split]
            for idx, ok in enumerate(legal, start=self.num_tables):
                action_mask[idx] = 1.0 if ok else 0.0

        return {
            "phase": phase,
            "balance": bal,
            "table_info": table_info,
            "in_hand": in_hand,
            "true_count": true_count,
            "action_mask": action_mask
        }

    def render(self, mode="human"):
        obs = self._get_observation()
        print("Obs:", obs)

    def close(self):
        pass


def env_test():
    env = CasinoEnv(num_tables=3)
    obs = env.reset()
    print("=== After Reset (Phase 0) ===")
    print("Observation:", obs)

    table_selection_action = 1
    obs, reward, done, info = env.step(table_selection_action)
    print("\n=== After Table Selection (Transition to Phase 1) ===")
    print("Observation:", obs)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)

    in_hand_action = 1
    obs, reward, done, info = env.step(in_hand_action)
    print("\n=== After In-Hand Decision (Phase 1 Complete) ===")
    print("Observation:", obs)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)

    print("\n=== Rendering Environment ===")
    env.render()
    env.close()


if __name__ == "__main__":
    env_test()
