# game.py

from random import shuffle, randint
from typing import Optional, List, Literal, Dict
import uuid


class Settings:
    maximum_players = 7
    number_of_decks = 8

    minimum_bet = 1
    maximum_bet = 10000
    blackjack_payout = 1.5

    allowed_shoe_change = True
    allowed_split = False
    allowed_insurance = False
    allowed_double_down = True

    shoe_cutting_card_position = (0.45, 0.55)
    shoe_cut_bounds = [
        int(52 * number_of_decks * shoe_cutting_card_position[0]),
        int(52 * number_of_decks * shoe_cutting_card_position[1])
    ]

    player_rank_value = {
        "ace": [1, 11],
        "king": 10,
        "queen": 10,
        "jack": 10,
        "10": 10,
        "9": 9,
        "8": 8,
        "7": 7,
        "6": 6,
        "5": 5,
        "4": 4,
        "3": 3,
        "2": 2,
    }

    dealer_rank_value = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "jack": 10,
        "queen": 10,
        "king": 10,
        "ace": 11
    }

    suits = [
        "spade",
        "club",
        "heart",
        "diamond"
    ]


class Card:
    def __init__(self, rank, suit=None, value=None):
        self.rank = rank
        self.suit = suit

        self.rank = rank
        self.suit = suit

        if rank == "CUT":
            self.value_hard = self.value_soft = 0
        elif rank == "ace":
            self.value_hard, self.value_soft = 11, 1
        else:
            if value is None:
                mapped = Settings.player_rank_value.get(rank)
                val = mapped if isinstance(mapped, (int, float)) else mapped[0]
                self.value_hard = self.value_soft = val
            else:
                self.value_hard = self.value_soft = value
        self.uuid = uuid.uuid4()

    def __eq__(self, other):
        if isinstance(other, Card):
            if self.suit and other.suit:
                return self.rank == other.rank and self.suit == other.suit
            return self.rank == other.rank
        return False

    def __repr__(self):
        return f"{self.rank} of {self.suit}s"

    def __str__(self):
        return f"{self.rank} of {self.suit}s"


class Hand:
    def __init__(self, role: Literal["player", "dealer"] = "player"):
        self.role = role

        self.bet = 0

        self.cards = []
        self.value_hard = 0
        self.value_soft = 0

        self.has_blackjack = False
        self.has_busted = False
        self.has_ace = False
        self.has_pair = False

        self.can_hit = True
        self.can_split = False
        self.can_double = False

        self.uuid = uuid.uuid4()

    def __eq__(self, other):
        if isinstance(other, Hand):
            return self.uuid == other.uuid
        return False

    def __str__(self):
        cards = ", ".join([str(card) for card in self.cards])
        info = "\n".join(f"{k}: {v}" for k, v in {
            "value_hard": self.value_hard,
            "value_soft": self.value_soft,
            "has_blackjack": self.has_blackjack,
            "has_busted": self.has_busted,
            "has_ace": self.has_ace,
            "has_pair": self.has_pair,
        }.items())
        return f"{cards}\n{info}"

    def __repr__(self):
        return f"Soft: {self.value_soft}, Hard: {self.value_hard}"

    def require_card(self):
        if len(self.cards) < 2:
            return True
        return False

    def pop_card(self, card):
        self.cards.remove(card)
        Logic.evaluate_hand(self)

    def add_card(self, card):
        self.cards.append(card)
        Logic.evaluate_hand(self)


class Player:
    def __init__(self, name: str = "John", balance: int = 100):
        self.player_id = uuid.uuid4()
        self.name = name
        self.balance = balance
        self.table_pos = None

        self.hands: List[Hand] = []

    def add_hand(self, hand: Hand = None) -> None:
        if hand is None:
            hand = Hand(role="player")
        self.hands.append(hand)

    def clear_hands(self) -> None:
        self.hands = []

    def place_bet(self, hand: Hand, amount: int) -> None:
        if not (Settings.minimum_bet <= amount <= Settings.maximum_bet):
            raise ValueError(f"Bet {amount} is out of allowed range.")

        if amount > self.balance:
            raise ValueError(f"Not enough balance.")

        hand.bet += amount
        self.balance -= amount

    def collect_winnings(self, winnings: float) -> None:
        self.balance += winnings


class Dealer:
    def __init__(self, name: str = "Dealer"):
        self.uuid = uuid.uuid4()
        self.name = name

        self.hand: Optional[Hand] = None

        self.upcard = None

    def start_new_hand(self) -> None:
        self.hand = Hand(role="dealer")
        self.upcard = None

    def dealer_logic(self, shoe) -> None:
        while True:
            if self.hand is None:
                break

            Logic.evaluate_hand(self.hand)

            if self.hand.value_hard < 17:
                card = shoe.deal()
                self.hand.add_card(card)
            else:
                break

    def __repr__(self):
        return f"Dealer(name={self.name}, hand={self.hand})"


class Shoe:
    PLUS_CARDS = [
        Card(rank='2'),
        Card(rank='3'),
        Card(rank='4'),
        Card(rank='5'),
        Card(rank='6')
    ]

    MINUS_CARDS = [
        Card(rank='10'),
        Card(rank='jack'),
        Card(rank='queen'),
        Card(rank='king'),
        Card(rank='ace')
    ]

    CUT_CARD = Card(rank='CUT', suit='CUT')

    def __init__(self):
        self.uuid = uuid.uuid4()

        self.shoe_cards = []
        self.dealt_cards = []
        self.deal_history = []

        self.reshuffle_required = False

        self.build()

    def true_count(self) -> float:
        return self.running_count() / Settings.number_of_decks

    def running_count(self) -> int:
        count = 0
        for card in self.dealt_cards:
            if card in self.PLUS_CARDS:
                count += 1
            elif card in self.MINUS_CARDS:
                count -= 1
            else:
                continue

        return count

    def build(self) -> None:
        self.reshuffle_required = False
        self.shoe_cards = [
            Card(rank=rank, suit=suit, value=value)
            for rank, value in Settings.player_rank_value.items() for suit in Settings.suits
            for _ in range(Settings.number_of_decks)
        ]
        shuffle(self.shoe_cards)
        self.shoe_cards.insert(randint(*Settings.shoe_cut_bounds), self.CUT_CARD)

    def deal(self, reshuffle_allowed: bool = False) -> Card:
        if self.reshuffle_required and reshuffle_allowed:
            self.build()
        card = self.shoe_cards.pop()
        if card == self.CUT_CARD:
            self.reshuffle_required = True
            card = self.shoe_cards.pop()
        self.dealt_cards.append(card)
        return card


class Table:
    def __init__(self,
                 shoe: Optional[Shoe] = None,
                 dealer: Optional[Dealer] = None):
        self.uuid = uuid.uuid4()

        self.shoe: Shoe = shoe or Shoe()
        self.dealer: Dealer = dealer or Dealer()
        self.players: Dict[int, Player or None] = {
            i: None for i in range(Settings.maximum_players)
        }
        self.seats: Dict[int, Literal["open", "occupied"] or None] = {
            i: "open" for i in range(Settings.maximum_players)
        }

    def seat_player(self, player: Player, index: int):
        self.players[index] = player
        self.seats[index] = "occupied"
        player.table_pos = index

    def remove_player(self, player: Player):
        self.players[player.table_pos] = None
        self.seats[player.table_pos] = "open"
        player.table_pos = None

    def start_new_round(self):
        self.dealer.start_new_hand()
        for idx, player in self.players.items():
            if player is not None:
                player.clear_hands()

    def __repr__(self):
        return f"Table(id={self.uuid}, players={len(self.players)})"


class Logic:
    @staticmethod
    def evaluate_hand(hand: Hand):
        ace_count = 0
        hard_value = 0
        soft_value = 0

        for card in hand.cards:
            hard_value += card.value_hard
            soft_value += card.value_soft
            if card.rank == "ace":
                ace_count += 1

        while hard_value > 21 and ace_count > 0:
            hard_value -= 10
            ace_count -= 1

        if ace_count > 0 and soft_value <= 21:
            hand.value_hard = hard_value
            hand.value_soft = soft_value
        else:
            hand.value_hard = hard_value
            hand.value_soft = hard_value

        hand.has_ace = (ace_count > 0)
        hand.has_pair = len(hand.cards) == 2 and hand.cards[0].rank == hand.cards[1].rank
        hand.has_blackjack = len(hand.cards) == 2 and hand.value_hard == 21
        hand.has_busted = hand.value_hard > 21

        hand.can_hit = not hand.has_blackjack and not hand.has_busted and hand.value_hard != 21
        if hand.role == "player":
            hand.can_split = len(hand.cards) == 2 and hand.cards[0].rank == hand.cards[1].rank
            hand.can_double = len(hand.cards) == 2 and not hand.has_blackjack

    @staticmethod
    def judge_hands(player_hand: Hand, dealer_hand: Hand) -> float:
        player_value = player_hand.value_hard
        dealer_value = dealer_hand.value_hard

        if player_hand.has_blackjack and not dealer_hand.has_blackjack:
            return player_hand.bet * (1 + Settings.blackjack_payout)
        if dealer_hand.has_blackjack and not player_hand.has_blackjack:
            return 0
        if player_hand.has_blackjack and dealer_hand.has_blackjack:
            return player_hand.bet
        if player_hand.has_busted:
            return 0
        if dealer_hand.has_busted:
            return player_hand.bet * 2
        if player_value > dealer_value:
            return player_hand.bet * 2
        if player_value < dealer_value:
            return 0
        return 0

    @staticmethod
    def judge_hands_norm(player_hand: Hand, dealer_hand: Hand) -> float:
        player_value = player_hand.value_hard
        dealer_value = dealer_hand.value_hard

        if player_hand.has_blackjack and not dealer_hand.has_blackjack:
            return 1.5
        if dealer_hand.has_blackjack and not player_hand.has_blackjack:
            return -1.0
        if player_hand.has_blackjack and dealer_hand.has_blackjack:
            return 0.0
        if player_hand.has_busted:
            return -1.0
        if dealer_hand.has_busted:
            return 1.0
        if player_value > dealer_value:
            return 1.0
        if player_value < dealer_value:
            return -1.0
        return 0.0


class Game:
    def __init__(self, table: Table):
        self.table = table

    def play_round(self):
        self.table.start_new_round()
        self.take_bets()
        self.deal_initial_cards()
        self.player_actions()
        self.dealer_turn()
        self.resolve_bets()

    def take_bets(self):
        for _, player in self.table.players.items():
            if player is not None:
                hand = Hand(role="player")
                player.add_hand(hand)
                player.place_bet(hand, player.balance // 10)

    def deal_initial_cards(self):
        for _ in range(2):
            for _, player in self.table.players.items():
                if player is not None:
                    for hand in player.hands:
                        card = self.table.shoe.deal()
                        hand.add_card(card)

            card = self.table.shoe.deal()
            self.table.dealer.hand.add_card(card)
            if self.table.dealer.upcard is None:
                self.table.dealer.upcard = card

    def player_actions(self):
        from basic_strategy import BasicStrategy
        strategy = BasicStrategy()

        for _, player in self.table.players.items():
            if player is None:
                continue

            for hand in player.hands:
                while hand.can_hit:
                    dealer_upcard = self.table.dealer.upcard
                    action = strategy.get_action(hand, dealer_upcard)

                    if action == "hit":
                        card = self.table.shoe.deal()
                        hand.add_card(card)

                    elif action == "stand":
                        break

                    elif action == "double" and hand.can_double:
                        player.place_bet(hand, hand.bet)  # Double the bet
                        card = self.table.shoe.deal()
                        hand.add_card(card)
                        break

                    elif action == "split":
                        break
                    else:
                        break

    def dealer_turn(self):
        if any(
                not hand.has_busted
                for p in self.table.players.values() if p is not None
                for hand in p.hands
        ):
            self.table.dealer.dealer_logic(self.table.shoe)

    def resolve_bets(self):
        dealer_hand = self.table.dealer.hand
        for _, player in self.table.players.items():
            if player is None:
                continue

            for hand in player.hands:
                winnings = Logic.judge_hands(hand, dealer_hand)
                if winnings > 0:
                    player.collect_winnings(winnings)

    def __repr__(self):
        return f"Game with table={self.table}"


def game_test():
    table = Table()
    player1 = Player(name="Alice", balance=500)
    player2 = Player(name="Bob", balance=300)
    table.seat_player(player1, 0)
    table.seat_player(player2, 1)
    game = Game(table)
    game.play_round()

    print(f"Alice's final balance: {player1.balance}")
    print(f"Alice's final hand: \n{player1.hands[0]}\n")
    print(f"Bob's final balance: {player2.balance}")
    print(f"Bob's final hand: \n{player2.hands[0]}\n")
    print(f"Dealer's final hand: {table.dealer.hand}")
    print(f"Dealer's final value: Soft {table.dealer.hand.value_soft} / Hard {table.dealer.hand.value_hard}")


if __name__ == "__main__":
    game_test()
