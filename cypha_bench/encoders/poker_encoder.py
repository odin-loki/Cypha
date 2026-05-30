from __future__ import annotations

import numpy as np


class PokerEncoder:
    """Texas Hold'em situation encoder with optional treys support."""

    def __init__(self) -> None:
        self._evaluator = None
        try:
            from treys import Card, Deck, Evaluator

            self._Card = Card
            self._Deck = Deck
            self._evaluator = Evaluator()
        except ImportError:
            self._Card = None
            self._Deck = None

    def encode_hand_situation(
        self,
        hole_cards,
        community_cards,
        pot_size: float,
        stack_size: float,
        position: int,
    ) -> np.ndarray:
        if self._evaluator is None:
            return self._synthetic_vector(pot_size, stack_size, position)

        Card = self._Card
        features: list[float] = []
        if len(community_cards) >= 3:
            rank = self._evaluator.evaluate(community_cards, hole_cards)
            hand_strength = 1.0 - (rank / 7462.0)
        else:
            hand_strength = self._preflop_strength(hole_cards)
        features.append(hand_strength)

        one_hot = [0.0] * 9
        if len(community_cards) >= 3:
            hand_class = self._evaluator.get_rank_class(
                self._evaluator.evaluate(community_cards, hole_cards)
            )
            one_hot[max(0, min(8, hand_class - 1))] = 1.0
        else:
            one_hot[8] = 1.0
        features.extend(one_hot)

        for card in hole_cards[:2]:
            features.append((Card.get_rank_int(card) - 2) / 12.0)
        if len(hole_cards) < 2:
            features.append(0.0)

        suited = 0.0
        if len(hole_cards) >= 2:
            suited = float(Card.get_suit_int(hole_cards[0]) == Card.get_suit_int(hole_cards[1]))
        features.append(suited)
        features.append(len(community_cards) / 5.0)
        features.append(min(pot_size, 10.0) / 10.0)
        features.append(min(stack_size, 10.0) / 10.0)
        features.append(float(position) / 2.0)
        features.append(self._estimate_outs(hole_cards, community_cards) / 47.0)
        return np.asarray(features[:18], dtype=np.float32)

    def _preflop_strength(self, hole_cards) -> float:
        Card = self._Card
        r1 = Card.get_rank_int(hole_cards[0])
        r2 = Card.get_rank_int(hole_cards[1])
        high = max(r1, r2) / 12.0
        gap = abs(r1 - r2) / 12.0
        suited = float(Card.get_suit_int(hole_cards[0]) == Card.get_suit_int(hole_cards[1]))
        return float(np.clip(high - 0.3 * gap + 0.1 * suited, 0.0, 1.0))

    def _estimate_outs(self, hole_cards, community_cards) -> float:
        if len(community_cards) < 3 or self._Card is None:
            return 0.0
        Card = self._Card
        h_suits = [Card.get_suit_int(c) for c in hole_cards]
        c_suits = [Card.get_suit_int(c) for c in community_cards]
        for suit in set(h_suits):
            if h_suits.count(suit) == 2 and c_suits.count(suit) >= 2:
                return 9.0
        return 2.0

    def generate_random_situation(self, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        if self._Deck is None:
            vec = self._synthetic_vector(
                pot=float(rng.integers(1, 50)) / 100.0,
                stack=float(rng.integers(10, 200)) / 200.0,
                position=int(rng.integers(0, 3)),
            )
            hs = float(vec[0])
            label = "raise" if hs > 0.65 else ("call" if hs > 0.35 else "fold")
            return vec, label

        deck = self._Deck()
        deck.shuffle()
        hole = deck.draw(2)
        flop = deck.draw(3)
        community = flop if rng.random() > 0.5 else flop + deck.draw(1)
        pot = float(rng.integers(1, 50)) / 100.0
        stack = float(rng.integers(10, 200)) / 200.0
        pos = int(rng.integers(0, 3))
        vec = self.encode_hand_situation(hole, community, pot, stack, pos)
        hs = float(vec[0])
        label = "raise" if hs > 0.65 else ("call" if hs > 0.35 else "fold")
        return vec, label

    def _synthetic_vector(self, pot: float, stack: float, position: int) -> np.ndarray:
        rng = np.random.default_rng(int(pot * 1000 + stack * 100 + position))
        hs = float(rng.uniform(0.0, 1.0))
        vec = np.zeros(18, dtype=np.float32)
        vec[0] = hs
        vec[8 + int(hs * 8)] = 1.0
        vec[11:14] = [rng.uniform(0, 1), rng.uniform(0, 1), rng.random()]
        vec[14] = pot
        vec[15] = stack
        vec[16] = position / 2.0
        vec[17] = rng.uniform(0, 0.2)
        return vec
