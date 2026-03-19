import random
from collections import Counter
import itertools

class Card:
    SUITS = ('S', 'H', 'D', 'C')
    # S: Spades, H: Hearts, D: Diamonds, C: Clubs
    RANKS = ('2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A')

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}{self.suit}"

class Deck:
    def __init__(self):
        self.cards = [Card(r, s) for r in Card.RANKS for s in Card.SUITS]
        random.shuffle(self.cards)

    def deal(self, num):
        return [self.cards.pop() for _ in range(num)]

class PokerGame:
    def __init__(self, player_names):
        self.deck = Deck()
        self.players = {name: [] for name in player_names}
        self.community_cards = []
        self.pot = 0

    def play_game(self):
        # 1. Deal Hole Cards
        for name in self.players:
            self.players[name] = self.deck.deal(2)
        
        print(f"Hole cards dealt. Players: {self.players}")

        # --- ROUND 1: PRE-FLOP ---
        self.betting_round("Pre-Flop")

        # 2. The Flop (3 cards)
        self.community_cards.extend(self.deck.deal(3))
        print(f"Flop: {self.community_cards}")

        # --- ROUND 2: POST-FLOP ---
        self.betting_round("Flop")

        # 3. The Turn (1 card)
        self.community_cards.extend(self.deck.deal(1))
        print(f"Turn: {self.community_cards}")

        # --- ROUND 3: POST-TURN ---
        self.betting_round("Turn")

        # 4. The River (1 card)
        self.community_cards.extend(self.deck.deal(1))
        print(f"River: {self.community_cards}")

        # --- ROUND 4: POST-RIVER ---
        self.betting_round("River")

        self.showdown()

    def betting_round(self, stage):
        """
        Placeholder for your custom betting logic.
        This is where you'll implement calls, raises, and folds.
        """
        print(f"\n--- Entering {stage} Betting Round ---")
        # TODO: Implement your custom logic here (e.g., player input, AI decisions)
        input("Press Enter to continue to next stage...")

    def showdown(self):
        print("\n--- Showdown ---")
        print(f"Community Cards: {self.community_cards}")
        for player, hand in self.players.items():
            print(f"{player}'s Hand: {hand}")

        # Compute total order and announce winner or tie
        groups, ordered_names, ranks_map = self.compute_total_order()
        if not ordered_names:
            print("No players to evaluate.")
            return
        if len(groups[0]) > 1:
            print("It's a tie between:", groups[0])
        else:
            winner_name = ordered_names[0]
            winner_rank = ranks_map[winner_name]
            print(f"Winner: {winner_name} with hand rank {winner_rank}")

    def _hand_rank(self, cards):
        """Return a tuple (rank_value, high_cards) for a 5-card hand."""
        ranks_str = Card.RANKS
        rank_map = {r: i for i, r in enumerate(ranks_str, 2)}
        cards = sorted(cards, key=lambda c: rank_map[c.rank], reverse=True)
        ranks = [c.rank for c in cards]
        suits = [c.suit for c in cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        counts = sorted(rank_counts.values(), reverse=True)
        unique_ranks = sorted(set([rank_map[r] for r in ranks]), reverse=True)

        # Check for flush
        flush = None
        for suit, count in suit_counts.items():
            if count >= 5:
                flush = [c for c in cards if c.suit == suit][:5]
                break

        # Check for straight
        def get_straight(ranks_list):
            ranks_set = set([rank_map[r] for r in ranks_list])
            for start in range(14, 5, -1):
                straight = set(range(start, start-5, -1))
                if straight <= ranks_set:
                    return start
            # Special case: A-2-3-4-5
            if set([14, 2, 3, 4, 5]) <= ranks_set:
                return 5
            return None

        straight_high = get_straight(ranks)
        straight_flush_high = None
        if flush:
            flush_ranks = [c.rank for c in flush]
            straight_flush_high = get_straight(flush_ranks)

        # Straight flush
        if straight_flush_high:
            return (8, [straight_flush_high])
        # Four of a kind
        if counts and counts[0] == 4:
            four = [rank_map[r] for r, c in rank_counts.items() if c == 4][0]
            kicker = max([rank_map[r] for r, c in rank_counts.items() if c != 4])
            return (7, [four, kicker])
        # Full house
        if counts and counts[0] == 3 and len(counts) > 1 and counts[1] >= 2:
            three = [rank_map[r] for r, c in rank_counts.items() if c == 3][0]
            two = max([rank_map[r] for r, c in rank_counts.items() if c == 2])
            return (6, [three, two])
        # Flush
        if flush:
            return (5, [rank_map[c.rank] for c in flush])
        # Straight
        if straight_high:
            return (4, [straight_high])
        # Three of a kind
        if counts and counts[0] == 3:
            three = [rank_map[r] for r, c in rank_counts.items() if c == 3][0]
            kickers = sorted([rank_map[r] for r, c in rank_counts.items() if c == 1], reverse=True)[:2]
            return (3, [three] + kickers)
        # Two pair
        if len(counts) > 1 and counts[0] == 2 and counts[1] == 2:
            pairs = sorted([rank_map[r] for r, c in rank_counts.items() if c == 2], reverse=True)
            kicker = max([rank_map[r] for r, c in rank_counts.items() if c == 1])
            return (2, pairs + [kicker])
        # One pair
        if counts and counts[0] == 2:
            pair = [rank_map[r] for r, c in rank_counts.items() if c == 2][0]
            kickers = sorted([rank_map[r] for r, c in rank_counts.items() if c == 1], reverse=True)[:3]
            return (1, [pair] + kickers)
        # High card
        return (0, unique_ranks[:5])

    def best_hand_rank(self, player_hand):
        """Return the best 5-card hand rank for a player's hole cards plus community."""
        all_cards = player_hand + self.community_cards
        best_rank = (-1, [])
        for combo in itertools.combinations(all_cards, 5):
            rank = self._hand_rank(combo)
            if rank > best_rank:
                best_rank = rank
        return best_rank

    def compute_total_order(self):
        """Compute total order of players based on best hand ranks.

        Returns:
            groups (list[list[str]]): list of groups of player names, each group contains players tied for the same rank, in descending order.
            ordered_names (list[str]): flattened list of player names ordered by groups.
            ranks_map (dict): mapping player name -> best_rank tuple.
        """
        player_names = list(self.players.keys())
        best = []
        ranks_map = {}
        for name in player_names:
            best_rank = self.best_hand_rank(self.players[name])
            ranks_map[name] = best_rank
            best.append((best_rank, name))

        best.sort(reverse=True)
        groups = []
        current_group = []
        last_rank = None
        for rank, name in best:
            if last_rank is None or rank != last_rank:
                if current_group:
                    groups.append(current_group)
                current_group = [name]
                last_rank = rank
            else:
                current_group.append(name)
        if current_group:
            groups.append(current_group)

        ordered_names = [n for grp in groups for n in grp]
        return groups, ordered_names, ranks_map

    def compare_players(self, permutation):
        """Check whether `permutation` matches the computed total order.

        Args:
            permutation (list[str]): a permutation (ordering) of player names.

        Returns:
            (matches: bool, computed_order: list[str])
                - matches: True if the permutation fits the computed total order (respecting tie groups), False otherwise.
                - computed_order: the flattened list of player names ordered by their computed ranks (ties grouped together).
        """
        # Validate permutation
        players_set = set(self.players.keys())
        perm_set = set(permutation)
        if players_set != perm_set:
            raise ValueError("Permutation must contain exactly the same player names")

        groups, ordered_names, _ = self.compute_total_order()

        # Check that permutation is composed of the groups in order, where members of each group
        # appear contiguously in any order.
        i = 0
        n = len(permutation)
        for grp in groups:
            gsize = len(grp)
            chunk = permutation[i:i+gsize]
            if set(chunk) != set(grp):
                return False, ordered_names
            i += gsize
        return (i == n), ordered_names
