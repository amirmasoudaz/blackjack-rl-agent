# basic_strategy.py

DEALER_RANK_VALUE = {
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


class BasicStrategy:
    def get_action(self, hand, dealer_upcard):
        if dealer_upcard is None:
            return "hit"
        dealer_value = DEALER_RANK_VALUE.get(dealer_upcard.rank, 10)
        if hand.has_pair and len(hand.cards) == 2:
            return self._pair_strategy(hand, dealer_value)
        if hand.has_ace and len(hand.cards) == 2 and not hand.has_pair:
            return self._soft_strategy(hand, dealer_value)
        return self._hard_strategy(hand, dealer_value)

    def _pair_strategy(self, hand, dealer_value):
        rank1 = hand.cards[0].rank
        rank2 = hand.cards[1].rank

        if rank1 != rank2:
            return "hit"

        # "ace" -> 11, "10"/"jack"/"queen"/"king" -> 10
        pair_value = self._rank_to_pair_value(rank1)

        # A,A -> always split
        if pair_value == 11:
            return "split"

        # 10,10 -> never split -> stand
        if pair_value == 10:
            return "stand"

        # 9,9 -> split vs 2-9 except 7, else stand
        if pair_value == 9:
            if 2 <= dealer_value <= 9 and dealer_value != 7:
                return "split"
            else:
                return "stand"

        # 8,8 -> always split
        if pair_value == 8:
            return "split"

        # 7,7 -> split vs 2-7, else hit
        if pair_value == 7:
            if 2 <= dealer_value <= 7:
                return "split"
            else:
                return "hit"

        # 6,6 -> split vs 2-6, else hit
        if pair_value == 6:
            if 2 <= dealer_value <= 6:
                return "split"
            else:
                return "hit"

        # 5,5 -> double vs 2-9, else hit
        if pair_value == 5:
            if 2 <= dealer_value <= 9:
                return "double"
            else:
                return "hit"

        # 4,4 -> split vs 5-6, else hit
        if pair_value == 4:
            if dealer_value in [5, 6]:
                return "split"
            else:
                return "hit"

        # 3,3 -> split vs 2-7, else hit
        if pair_value == 3:
            if 2 <= dealer_value <= 7:
                return "split"
            else:
                return "hit"

        # 2,2 -> split vs 2-7, else hit
        if pair_value == 2:
            if 2 <= dealer_value <= 7:
                return "split"
            else:
                return "hit"
        return "hit"

    def _soft_strategy(self, hand, dealer_value):
        card_values = []
        for c in hand.cards:
            if c.rank == "ace":
                card_values.append(11)
            elif c.rank in ["10", "jack", "queen", "king"]:
                card_values.append(10)
            else:
                card_values.append(int(c.rank))

        soft_value = sum(card_values)

        if soft_value == 20:  # A,9
            return "stand"
        elif soft_value == 19:  # A,8
            if dealer_value == 6:
                return "double"
            else:
                return "stand"
        elif soft_value == 18:  # A,7
            if 2 <= dealer_value <= 6:
                return "double"
            elif dealer_value in [9, 10, 11]:
                return "hit"
            else:
                return "stand"  # dealer 7 or 8
        elif soft_value == 17:  # A,6
            if 3 <= dealer_value <= 6:
                return "double"
            else:
                return "hit"
        elif soft_value == 16:  # A,5
            if 4 <= dealer_value <= 6:
                return "double"
            else:
                return "hit"
        elif soft_value == 15:  # A,4
            if 4 <= dealer_value <= 6:
                return "double"
            else:
                return "hit"
        elif soft_value == 14:  # A,3
            if 5 <= dealer_value <= 6:
                return "double"
            else:
                return "hit"
        elif soft_value == 13:  # A,2
            if 5 <= dealer_value <= 6:
                return "double"
            else:
                return "hit"
        else:
            return "hit"

    def _hard_strategy(self, hand, dealer_value):
        total = 0
        for c in hand.cards:
            if c.rank in ["jack", "queen", "king"]:
                total += 10
            elif c.rank == "ace":
                total += 1
            else:
                total += int(c.rank)

        if total >= 17:
            return "stand"
        elif total == 16:
            if 2 <= dealer_value <= 6:
                return "stand"
            else:
                return "hit"
        elif total == 15:
            if 2 <= dealer_value <= 6:
                return "stand"
            else:
                return "hit"
        elif total == 14:
            if 2 <= dealer_value <= 6:
                return "stand"
            else:
                return "hit"
        elif total == 13:
            if 2 <= dealer_value <= 6:
                return "stand"
            else:
                return "hit"
        elif total == 12:
            if 4 <= dealer_value <= 6:
                return "stand"
            else:
                return "hit"
        elif total == 11:
            return "double"
        elif total == 10:
            if 2 <= dealer_value <= 9:
                return "double"
            else:
                return "hit"
        elif total == 9:
            if 3 <= dealer_value <= 6:
                return "double"
            else:
                return "hit"
        else:
            return "hit"

    @staticmethod
    def _rank_to_pair_value(rank_str):
        if rank_str == "ace":
            return 11
        elif rank_str in ["10", "jack", "queen", "king"]:
            return 10
        else:
            return int(rank_str)
