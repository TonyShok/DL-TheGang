import os
import math
import random
import itertools
import json
from collections import Counter

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# =========================
# 任务说明常量 (Instruction)
# =========================
TASK_INSTRUCTION = """# 任务说明：德州扑克位次博弈 (Poker Positioning Game)

## 1. 你的角色与目标
你是一个智能博弈代理。你正在与其他玩家进行一场**合作策略游戏**。
**你的目标**：调整自己的位次，使得游戏结束时，所有玩家构成的序列与其“手牌强度”完全符合。
* **索引 1 (最强)**：必须是全场手牌最强的玩家。
* **索引 N (最弱)**：必须是全场手牌最弱的玩家。
**胜负判定**：这是一种“全有或全无”的挑战。只有当全场所有人的顺序都符合“强度从强到弱”的排列时（如果你和别人的牌力**完全相同（平局）**，你们两个谁在前谁在后都可以，不影响满分奖励），所有人才能获得 **1.0** 奖励；否则，所有人获得 **0.0**。

## 2. 基础规则详解
由于你可能不熟悉德州扑克，请严格遵守以下规则：

### A. 牌面构成
牌面表示：每张牌由“数字+花色”组成（如 10H 代表红桃10，AS 代表黑桃A）。
数字 (Rank)：2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A（手牌大小按这个顺序递增, A 为最大, 2 为最小）。
花色 (Suit)：S (黑桃), H (红桃), D (方块), C (草花)。
* **手牌**：你手中有 2 张只有你能看见的底牌。
* **公共牌**：桌面上公开的牌（0 到 5 张），所有人共用。

### B. 牌型强度排列（从强到弱）
你需要用你的 2 张底牌与公共牌组合，选出最强的 5 张牌，按以下等级比大小：
1. **同花顺**：同花色的顺子。
2. **四条**：四张相同数字的牌。
3. **葫芦**：三张相同数字 + 一对。
4. **同花**：五张牌花色相同。
5. **顺子**：五张数字连续的牌（如 5-6-7-8-9）。
6. **三条**：三张相同数字的牌。
7. **两对**：两个对子。
8. **对子**：两张相同数字的牌。
9. **高牌**：没有任何组合，只比单张大小。

## 3. 核心策略：信息传递
这是一个**信息不对称**的游戏。
* 你只知道自己的牌，不知道别人的牌。
* **位次变动是唯一的信息来源**：如果一个玩家冒着风险冲向“索引 1”，说明他试图告诉你他的牌非常强。
* 你需要观察其他玩家的行为，结合你自己的牌力，猜测你应该处于什么位置。

## 4. 输出规范
* 你的输出必须且只能包含一个用 `\\boxed{}` 包裹的数字。
* 这个数字代表你想要移动到的**目标索引 (aim)**，范围是 1 到 N。
* **禁止输出任何解释文字。**

## 5. 决策示例

### 示例一（3人局，Pre-Flop阶段）
* **观察 (Observation)**：
    * `my_hand`: `[AS, AH]` (一对A，全场最强的起始底牌)
    * `community`: `[]` (尚未发牌)
    * `current_order`: `['C', 'B', 'A']` (你目前在最后一名)
    * `my_id`: `'A'` | `my_pos`: 3
* **分析**：你拥有一对A，极其强大，理应排在最前面（索引 1）。
* **输出**：`\\boxed{1}`

### 示例二（5人局，River阶段）
* **观察 (Observation)**：
    * `my_hand`: `[2S, 7D]` (数字很小，且不凑对，极弱)
    * `community`: `[KH, QH, JH, 10H, 2H]` (桌面上已经是同花)
    * `current_order`: `['A', 'B', 'C', 'D', 'E']`
    * `my_id`: `'A'` | `my_pos`: 1
* **分析**：虽然桌面上有强大的公共牌，但所有人都共用它。你的底牌非常差，完全无法增强牌力。此时你排在第 1 位（最强位），这会干扰其他真正有大牌的人。为了全场拿到 1.0 奖励，你应该主动退到最后。
* **输出**：`\\boxed{5}`

### 6. 输入格式 (Observation)
在每一轮决策时，你会获得以下信息：
* `hand`: 你的 2 张底牌。
* `community`: 当前已发出的社区牌。
* `current_order`: 玩家当前的实时位次序列。
* `current_pos`: 你当前所在的索引位置。
* `stage`: 当前比赛阶段（如 "Flop"）。
* `history`: 之前各阶段达成的最终位次快照。

### 7. 决策准则
1.  **评估实力**：根据当前已知牌面，估算你在所有玩家中的实力分位。
2.  **表达意图**：如果你认为你的实力被低估（排在比你弱的人后面），你应该通过 `aim` 向上移动。
3.  **达成共识**：如果当前位次已经正确反映了你的实力范围（包括平局情况），你应该选择保持当前位置，以尽快结束回合获取奖励。"""

# =========================
# 基础牌类
# =========================

class Card:
    SUITS = ("S", "H", "D", "C")
    RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}{self.suit}"

RANK_TO_VAL = {r: i for i, r in enumerate(Card.RANKS, 2)}
VAL_TO_RANK = {v: r for r, v in RANK_TO_VAL.items()}

class Deck:
    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.cards = [Card(r, s) for r in Card.RANKS for s in Card.SUITS]
        self.rng.shuffle(self.cards)

    def deal(self, num):
        return [self.cards.pop() for _ in range(num)]

# =========================
# 叙事型德州位次博弈
# =========================

class NarrativePokerGame:
    STAGES = ["Pre-Flop", "Flop", "Turn", "River"]

    def __init__(
        self,
        player_names,
        seed=None,
        max_small_rounds=4,
        assist_model=None,
        collect_training=False
    ):
        if seed is None:
            seed = random.SystemRandom().randrange(1, 10**18)

        self.seed = seed
        self.rng = random.Random(seed)
        self.deck = Deck(self.rng)

        self.player_names = list(player_names)
        self.players = {name: [] for name in player_names}
        self.community_cards = []

        self.current_order = list(player_names)
        self.rng.shuffle(self.current_order)

        self.history_orders = {}
        self.stage_logs = {}
        self.max_small_rounds = max_small_rounds

        self.assist_model = assist_model
        self.collect_training = collect_training
        self.training_samples = []

        self.current_stage_name = None
        self.current_small_round = None
        self.action_history = []
        
        self.stage_commit_pos = {p: 0 for p in self.player_names}
        self.stage_claim_count = {p: 0 for p in self.player_names}
        self.stage_reclaim_count = {p: 0 for p in self.player_names}
        self.stage_entry_private_info = {}
        
        self.log_buffer = []
        
        # 新增：用于存储 JSONL 数据的列表
        self.json_steps = []

    def log(self, text):
        self.log_buffer.append(text)

    # -------------------------
    # 展示辅助
    # -------------------------

    def cards_str(self, cards):
        return "[" + ", ".join(map(str, cards)) + "]"

    def sort_cards_desc(self, cards):
        return sorted(cards, key=lambda c: (RANK_TO_VAL[c.rank], c.suit), reverse=True)

    def val_to_rank(self, v):
        return VAL_TO_RANK[v]

    def stage_scene_text(self, stage):
        if stage == "Pre-Flop": return "现在是刚发完两张底牌后的 Pre-Flop 回合，还没有公共牌。"
        if stage == "Flop": return f"翻牌是 {self.cards_str(self.community_cards)}。"
        if stage == "Turn": return f"转牌后公共牌是 {self.cards_str(self.community_cards)}。"
        return f"河牌后公共牌已经定格为 {self.cards_str(self.community_cards)}。"

    def current_board_pair_rank(self):
        rank_counts = Counter(c.rank for c in self.community_cards)
        pair_vals = [RANK_TO_VAL[r] for r, cnt in rank_counts.items() if cnt >= 2]
        if not pair_vals: return None
        return max(pair_vals)

    # -------------------------
    # 牌型评估
    # -------------------------

    def _hand_rank(self, cards):
        cards = self.sort_cards_desc(cards)
        ranks = [c.rank for c in cards]
        suits = [c.suit for c in cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        counts = sorted(rank_counts.values(), reverse=True)
        unique_ranks = sorted(set(RANK_TO_VAL[r] for r in ranks), reverse=True)

        def get_straight(ranks_list):
            vals = set(RANK_TO_VAL[r] for r in ranks_list)
            for start in range(14, 5, -1):
                need = set(range(start, start - 5, -1))
                if need <= vals: return start
            if {14, 2, 3, 4, 5} <= vals: return 5
            return None

        flush = None
        for s, cnt in suit_counts.items():
            if cnt >= 5:
                flush = [c for c in cards if c.suit == s][:5]
                break

        straight_high = get_straight(ranks)
        straight_flush_high = None
        if flush: straight_flush_high = get_straight([c.rank for c in flush])

        if straight_flush_high: return (8, [straight_flush_high])
        if counts and counts[0] == 4:
            four = max(RANK_TO_VAL[r] for r, c in rank_counts.items() if c == 4)
            kicker = max(RANK_TO_VAL[r] for r, c in rank_counts.items() if c != 4)
            return (7, [four, kicker])
        if counts == [3, 2]:
            three = max(RANK_TO_VAL[r] for r, c in rank_counts.items() if c == 3)
            two = max(RANK_TO_VAL[r] for r, c in rank_counts.items() if c == 2)
            return (6, [three, two])
        if flush: return (5, [RANK_TO_VAL[c.rank] for c in flush])
        if straight_high: return (4, [straight_high])
        if counts and counts[0] == 3:
            three = max(RANK_TO_VAL[r] for r, c in rank_counts.items() if c == 3)
            kickers = sorted((RANK_TO_VAL[r] for r, c in rank_counts.items() if c == 1), reverse=True)[:2]
            return (3, [three] + kickers)
        if len(counts) > 1 and counts[0] == 2 and counts[1] == 2:
            pairs = sorted((RANK_TO_VAL[r] for r, c in rank_counts.items() if c == 2), reverse=True)
            kicker = max(RANK_TO_VAL[r] for r, c in rank_counts.items() if c == 1)
            return (2, pairs + [kicker])
        if counts and counts[0] == 2:
            pair = max(RANK_TO_VAL[r] for r, c in rank_counts.items() if c == 2)
            kickers = sorted((RANK_TO_VAL[r] for r, c in rank_counts.items() if c == 1), reverse=True)[:3]
            return (1, [pair] + kickers)
        return (0, unique_ranks[:5])

    def best_hand_detail(self, player_hand):
        all_cards = player_hand + self.community_cards
        best_rank = (-1, [])
        best_combo = None
        for combo in itertools.combinations(all_cards, 5 if len(all_cards)>=5 else len(all_cards)):
            if len(combo) == 5:
                rank = self._hand_rank(combo)
            else:
                ranks = sorted([RANK_TO_VAL[c.rank] for c in combo], reverse=True)
                rank = (1 if ranks[0]==ranks[1] else 0, ranks) if len(combo)==2 else (0, ranks)
            
            if rank > best_rank:
                best_rank = rank
                best_combo = combo
        return best_rank, self.sort_cards_desc(best_combo) if best_combo else []

    def rank_tuple_to_text(self, rank_tuple):
        rank_value, highs = rank_tuple
        if rank_value == 8: return f"同花顺（顶张 {self.val_to_rank(highs[0])}）"
        if rank_value == 7: return f"四条（{self.val_to_rank(highs[0])}）"
        if rank_value == 6: return f"葫芦（{self.val_to_rank(highs[0])} 带 {self.val_to_rank(highs[1])}）"
        if rank_value == 5: return f"同花"
        if rank_value == 4: return f"顺子"
        if rank_value == 3: return f"三条（{self.val_to_rank(highs[0])}）"
        if rank_value == 2: return f"两对（{self.val_to_rank(highs[0])} 和 {self.val_to_rank(highs[1])}）"
        if rank_value == 1: return f"一对（{self.val_to_rank(highs[0])}）"
        return f"高牌"

    # -------------------------
    # 绝对牌力评估
    # -------------------------

    def get_absolute_power(self, player, stage):
        hand = self.players[player]
        c1, c2 = self.sort_cards_desc(hand)
        v1, v2 = RANK_TO_VAL[c1.rank], RANK_TO_VAL[c2.rank]
        is_pair = (v1 == v2)
        is_suited = (c1.suit == c2.suit)

        if stage == "Pre-Flop":
            score = v1 * 14 + v2
            if is_pair: score += 2000
            if is_suited: score += 500
            return float(score)
        else:
            rank, best_combo = self.best_hand_detail(hand)
            rv, kickers = rank
            
            board_best, _ = self.best_hand_detail([])
            if rv == board_best[0] and kickers == board_best[1]:
                return 0.0 # 蹭局的空气，剥夺所有分数
            
            score = rv * 1000000
            for i, k in enumerate(kickers):
                score += k * (14**(4-i))
            return float(score)

    def compute_total_order(self):
        best = []
        detail_map = {}
        for name in self.player_names:
            rank, combo = self.best_hand_detail(self.players[name])
            detail_map[name] = {"rank": rank, "combo": combo}
            best.append((self.get_absolute_power(name, "River"), name))

        best.sort(reverse=True)
        groups, current_group, last_rank = [], [], None
        for rank, name in best:
            if last_rank is None or rank != last_rank:
                if current_group: groups.append(current_group)
                current_group = [name]
                last_rank = rank
            else:
                current_group.append(name)
        if current_group: groups.append(current_group)

        ordered_names = [n for grp in groups for n in grp]
        return groups, ordered_names, detail_map

    def compare_players(self, permutation):
        groups, ordered_names, _ = self.compute_total_order()
        i, n = 0, len(permutation)
        for grp in groups:
            gsize = len(grp)
            chunk = permutation[i:i + gsize]
            if set(chunk) != set(grp): return False, ordered_names
            i += gsize
        return (i == n), ordered_names

    def get_reward(self):
        matches, _ = self.compare_players(self.current_order)
        return 1.0 if matches else 0.0

    # -------------------------
    # 高级推断竞价逻辑 (Optimized Bidding)
    # -------------------------

    def _choose_action(self, player, stage):
        power = self.get_absolute_power(player, stage)
        current_pos = self.current_order.index(player)
        my_idx = self.player_names.index(player)
        
        # 1. 制定更细腻的初始锚点
        if stage == "Pre-Flop":
            if power >= 2000: base_aim = 0     # 所有的对子 AA-22
            elif power >= 500: base_aim = 1    # 优质同花
            elif power >= 120: base_aim = 2    # 高张非同花
            else: base_aim = 3                 # 垃圾牌
        else:
            if power >= 3000000: base_aim = 0  # 三条或以上坚果
            elif power >= 1500000: base_aim = 1# 两对及好踢脚的顶对
            elif power >= 500000: base_aim = 2 # 小对子
            else: base_aim = 3                 # 空气
            
        if self.current_small_round == 1:
            return base_aim
            
        # 2. 动态博弈逼退估值 (Tie-breaking Engine)
        opp_powers = []
        for p in self.player_names:
            if p != player:
                c_pos = self.stage_commit_pos[p]
                claims = self.stage_claim_count[p]
                reclaims = self.stage_reclaim_count.get(p, 0)
                agg = claims + reclaims # 强硬度
                
                est = 0
                if stage == "Pre-Flop":
                    if c_pos == 0: est = 2000 + agg * 60
                    elif c_pos == 1: est = 500 + agg * 60
                    elif c_pos == 2: est = 120 + agg * 30
                    else: est = agg * 30
                else:
                    if c_pos == 0: est = 3000000 + agg * 1000000
                    elif c_pos == 1: est = 1500000 + agg * 400000
                    elif c_pos == 2: est = 500000 + agg * 200000
                    else: est = agg * 100000
                    
                opp_powers.append((est, self.player_names.index(p)))
                
        # 严格消除平局死锁，匹配 compute_total_order 的 Python 原生字符比对降序规则
        my_rank = 0
        for est, opp_idx in opp_powers:
            if power < est:
                my_rank += 1
            elif power == est:
                # 破冰处理：如果估值完全相等（如全是0战力的空气），按固定的索引强行错开，绝不撞车
                if opp_idx > my_idx:
                    my_rank += 1
                    
        aim = my_rank
        
        # 3. 神经网络特征级干预 
        if not self.collect_training and self.assist_model is not None:
            best_aim = aim
            best_score = self._candidate_nn_score(player, stage, aim)
            for cand in range(4):
                if cand != aim:
                    sc = self._candidate_nn_score(player, stage, cand)
                    # 提升规则置信阈值至0.4，防止NN错误覆盖极其精妙的算力排序
                    if sc > best_score + 0.4: 
                        best_score = sc
                        best_aim = cand
            aim = best_aim

        # 4. 终局防抖机制
        if aim < current_pos and self.current_small_round >= self.max_small_rounds:
            aim = current_pos 

        return aim

    def _compose_thought(self, player, stage, current_pos, aim):
        power = self.get_absolute_power(player, stage)
        scene = self.stage_scene_text(stage)
        hole_text = f"我手里的底牌是 {self.cards_str(self.players[player])}。"
        
        if stage == "Pre-Flop":
            if power >= 2000: strength_text = "这是极具压制力的顶级口袋对，绝对宣示主权。"
            elif power >= 500: strength_text = "这是一手优质同花连牌或高张，理应去前排建立威慑。"
            elif power >= 120: strength_text = "这牌潜力尚可，适合待在中游。"
            else: strength_text = "这是极其垃圾的散牌，必须主动退让去垫底。"
        else:
            rank, _ = self.best_hand_detail(self.players[player])
            if power == 0.0:
                strength_text = "虽然公牌很大，但我底牌完全没用上，我是碰瓷蹭局的纯空气！"
            elif power >= 3000000:
                strength_text = f"我击中了强大的 {self.rank_tuple_to_text(rank)}，这是怪物坚果牌。"
            elif power >= 1500000:
                strength_text = f"我击中了扎实的 {self.rank_tuple_to_text(rank)}，压制力不错。"
            elif power >= 500000:
                strength_text = f"我仅有中低对子 {self.rank_tuple_to_text(rank)}，勉强自保。"
            else:
                strength_text = "我没有击中任何有效牌型，只能退让。"

        if aim < current_pos: action_text = f"综合局势与我的硬实力，我决定强硬前压到索引 {aim + 1}。"
        elif aim > current_pos: action_text = f"前排争抢太过凶猛，对比实力后我理智退让到索引 {aim + 1} 保全胜率。"
        else: action_text = f"我权衡当前局势，决定坚守在目前的索引 {aim + 1}。"
            
        return f"{scene} {hole_text} {strength_text} {action_text}"

    # -------------------------
    # 兼容训练特征的提取
    # -------------------------

    def _init_stage_info(self, stage):
        info = {}
        for p in self.player_names:
            rank, _ = self.best_hand_detail(self.players[p])
            suits = Counter(c.suit for c in self.players[p] + self.community_cards)
            flush_draw = any(v == 4 for v in suits.values())
            info[p] = {
                "rank": rank if stage != "Pre-Flop" else None,
                "draws": {"flush_draw": flush_draw, "open_ended": False, "gutshot": False}
            }
        return info

    def _feature_vector(self, player, stage):
        feat = []
        stage_idx = self.STAGES.index(stage)
        for i in range(len(self.STAGES)): feat.append(1.0 if i == stage_idx else 0.0)
        feat.append(len(self.community_cards) / 5.0)

        c1, c2 = self.sort_cards_desc(self.players[player])
        v1, v2 = RANK_TO_VAL[c1.rank], RANK_TO_VAL[c2.rank]
        feat.extend([v1 / 14.0, v2 / 14.0, 1.0 if c1.suit == c2.suit else 0.0, 1.0 if v1 == v2 else 0.0, abs(v1-v2)/12.0])
        feat.append(self.get_absolute_power(player, stage) / 2000000.0)

        info = self.stage_entry_private_info[player]
        if info["rank"] is None:
            feat.extend([0.0] * 17)
        else:
            rv, highs = info["rank"]
            for i in range(9): feat.append(1.0 if i == rv else 0.0)
            padded = list(highs) + [0] * (5 - len(highs))
            feat.extend([x / 14.0 for x in padded[:5]])
            draws = info["draws"]
            feat.append(1.0 if draws["flush_draw"] else 0.0)
            feat.append(1.0 if draws["open_ended"] else 0.0)
            feat.append(1.0 if draws["gutshot"] else 0.0)

        rank_counts = Counter(c.rank for c in self.community_cards)
        for r in Card.RANKS: feat.append(rank_counts.get(r, 0) / 5.0)
        suit_counts = Counter(c.suit for c in self.community_cards)
        for s in Card.SUITS: feat.append(suit_counts.get(s, 0) / 5.0)

        pair_flag = 1.0 if any(cnt >= 2 for cnt in rank_counts.values()) else 0.0
        trips_flag = 1.0 if any(cnt >= 3 for cnt in rank_counts.values()) else 0.0
        three_same_suit = 1.0 if suit_counts and max(suit_counts.values()) >= 3 else 0.0
        board_pair_rank = self.current_board_pair_rank()
        feat.extend([pair_flag, trips_flag, three_same_suit, 0.0, 0.0 if board_pair_rank is None else board_pair_rank/14.0])

        pos_now = {p: self.current_order.index(p) for p in self.player_names}
        for p in self.player_names: feat.append(pos_now[p] / 3.0)
        for p in self.player_names: feat.append(pos_now[p] / 3.0)
        for p in self.player_names: feat.append(pos_now[p] / 3.0)

        for p in self.player_names:
            feat.extend([self.stage_claim_count[p]/4.0, 0.0, self.stage_reclaim_count[p]/4.0, 0.0, 0.0, pos_now[p]/3.0])

        for p in self.player_names: feat.append(self.get_absolute_power(p, stage) / 2000000.0)
        for q in self.player_names:
            if q != player: feat.extend([0.0, 0.0])

        while len(feat) < 96: feat.append(0.0)
        return feat[:96]

    def _candidate_nn_score(self, player, stage, candidate_pos):
        if self.assist_model is None: return 0.0
        feat = self._feature_vector(player, stage)
        probs = self.assist_model.predict(feat)
        prob_map = {p: probs[i] for i, p in enumerate(self.player_names) if p != player}

        others = [p for p in self.current_order if p != player]
        tentative = others[:candidate_pos] + [player] + others[candidate_pos:]

        score = 0.0
        eps = 1e-4
        for idx, q in enumerate(tentative):
            if q == player: continue
            p = min(max(prob_map[q], eps), 1.0 - eps)
            if candidate_pos < idx: score += math.log(p)
            else: score += math.log(1.0 - p)
        return score

    def _record_training_sample(self, player, stage):
        if not self.collect_training: return
        x = self._feature_vector(player, stage)
        power_map = {p: self.get_absolute_power(p, stage) for p in self.player_names}
        y, mask = [], []
        eps = 1.0
        for p in self.player_names:
            if p == player:
                y.append(0.5); mask.append(0.0)
            else:
                if power_map[player] > power_map[p] + eps: y.append(1.0)
                elif power_map[player] < power_map[p] - eps: y.append(0.0)
                else: y.append(0.5)
                mask.append(1.0)
        self.training_samples.append((x, y, mask))

    # -------------------------
    # 动作执行循环
    # -------------------------

    def betting_round(self, stage):
        self.current_stage_name = stage
        self.stage_entry_private_info = self._init_stage_info(stage)
        self.stage_commit_pos = {p: self.current_order.index(p) for p in self.player_names}
        self.stage_claim_count = {p: 0 for p in self.player_names}
        self.stage_reclaim_count = {p: 0 for p in self.player_names}
        
        self.log(f"#### {stage}")
        self.log(f"- 当前公共牌: {['[' + ', '.join([str(c) for c in self.community_cards]) + ']'] if self.community_cards else '[]'}")
        self.log(f"- 进入本轮前位次: {self.current_order}")

        speak_order = list(reversed(self.current_order))
        self.log(f"- 本轮固定发言顺序（后位先说，整轮保持不变）: {speak_order}")

        for sr in range(1, self.max_small_rounds + 1):
            self.current_small_round = sr
            self.log(f"- Small Round {sr} 开始位次: {self.current_order}")
            any_changes = False

            for player in speak_order:
                pos = self.current_order.index(player)
                self._record_training_sample(player, stage)
                
                # 构建符合要求的 Input 字符串
                current_input = (
                    f"current_order：{self.current_order},history：{self.history_orders}\n"
                    f"current_pos：{pos + 1}，hand：{self.cards_str(self.players[player])}\n"
                    f"community：{self.cards_str(self.community_cards)}，stage：{stage}"
                )

                aim = self._choose_action(player, stage)
                thought = self._compose_thought(player, stage, pos, aim)
                
                # 构建输出：决策需要转为 1-N 的 1-indexed 形式
                current_output = f"{thought}\n决策动作：\\boxed{{{aim + 1}}}"
                
                self.json_steps.append([current_input, current_output])
                
                self.log(f"  - {player} 视角独白：{thought}")
                self.log(f"    决策动作：\\boxed{{{aim + 1}}}（索引 {pos + 1} -> {aim + 1}）")

                if aim != pos:
                    if aim < pos:
                        if self.stage_commit_pos[player] <= aim:
                            self.stage_reclaim_count[player] += 1
                        self.stage_claim_count[player] += 1
                        
                    p_obj = self.current_order.pop(pos)
                    self.current_order.insert(aim, p_obj)
                    any_changes = True
                    self.stage_commit_pos[player] = aim

            self.log(f"  - Small Round {sr} 结束位次: {self.current_order}")
            if not any_changes: break

        self.history_orders[stage] = list(self.current_order)
        self.log(f"- 本轮形成的共识位次: {self.current_order}\n")

    def play_game(self):
        for name in self.player_names:
            self.players[name] = self.deck.deal(2)
        self.betting_round("Pre-Flop")
        self.community_cards.extend(self.deck.deal(3))
        self.betting_round("Flop")
        self.community_cards.extend(self.deck.deal(1))
        self.betting_round("Turn")
        self.community_cards.extend(self.deck.deal(1))
        self.betting_round("River")

    def to_jsonl_dict(self):
        if not self.json_steps:
            return None
        main_input, main_output = self.json_steps[-1]
        history_steps = self.json_steps[:-1]
        return {
            "instruction": TASK_INSTRUCTION,
            "input": main_input,
            "output": main_output,
            "history": history_steps
        }

    def render_report(self, game_idx=1):
        lines = [f"### 对局 [{game_idx}]", f"[随机种子: {self.seed}]", "[真实底牌汇总，仅供最终结算参考；玩家思维链中不允许互相透底]"]
        for p in self.player_names: lines.append(f"- {p}: {self.cards_str(self.players[p])}")
        lines.extend(self.log_buffer)
        
        groups, ordered_names, detail_map = self.compute_total_order()
        lines.append("[结算结果]")
        lines.append("#### 最终 Showdown")
        lines.append(f"- 最终公共牌: {['[' + ', '.join([str(c) for c in self.community_cards]) + ']'] if self.community_cards else '[]'}")
        
        for name in ordered_names:
            rank = detail_map[name]["rank"]
            combo = detail_map[name]["combo"]
            lines.append(f"- {name}: 底牌 {self.cards_str(self.players[name])}；最强组合 {self.cards_str(combo)}；牌型 {self.rank_tuple_to_text(rank)}")
            
        lines.append(f"- 博弈共识位次: {self.current_order}")
        lines.append(f"- 真实实力分组: {groups}")
        lines.append(f"- 最终奖励 (Reward): {self.get_reward()}")
        return "\n".join(lines)


# =========================
# 神经网络：CPU 辅助模型
# =========================

class PairwiseAssistNet(nn.Module):
    def __init__(self, input_dim, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class NeuralAssistWrapper:
    def __init__(self, model, player_names, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.player_names = list(player_names)
        self.device = torch.device(device)

    def predict(self, feature_vec):
        x = torch.tensor(feature_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)[0].cpu().tolist()
        return probs

def masked_bce_with_logits(logits, targets, mask):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
    return loss

def collect_supervised_samples(num_games=1500, max_small_rounds=4):
    players = ["A", "B", "C", "D"]
    samples = []
    for i in range(num_games):
        game = NarrativePokerGame(player_names=players, max_small_rounds=max_small_rounds, collect_training=True)
        game.play_game()
        samples.extend(game.training_samples)
        if (i + 1) % 300 == 0:
            print(f"[collect] 已生成纯净训练对局 {i + 1}/{num_games}，样本数 {len(samples)}")
    return samples

def train_cpu_assist_model(model_path="pairwise_assist_cpu.pt", num_games=1500, train_epochs=5, batch_size=1024, lr=1e-3):
    if not TORCH_AVAILABLE: raise RuntimeError("未检测到 torch")

    print("[train] 开始采样高质量零噪声监督数据...")
    samples = collect_supervised_samples(num_games=num_games, max_small_rounds=4)

    xs = torch.tensor([s[0] for s in samples], dtype=torch.float32)
    ys = torch.tensor([s[1] for s in samples], dtype=torch.float32)
    ms = torch.tensor([s[2] for s in samples], dtype=torch.float32)

    n = xs.shape[0]
    input_dim = xs.shape[1]
    perm = torch.randperm(n)
    split = int(n * 0.9)

    train_ds = TensorDataset(xs[perm[:split]], ys[perm[:split]], ms[perm[:split]])
    val_ds = TensorDataset(xs[perm[split:]], ys[perm[split:]], ms[perm[split:]])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PairwiseAssistNet(input_dim=input_dim, output_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"[train] 样本总数: {n}, 输入维度: {input_dim}")
    print("[train] 开始 CPU 训练...")

    for ep in range(train_epochs):
        model.train()
        train_loss_sum = 0.0
        for xb, yb, mb in train_loader:
            optimizer.zero_grad()
            loss = masked_bce_with_logits(model(xb), yb, mb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                val_loss_sum += masked_bce_with_logits(model(xb), yb, mb).item()

        print(f"[train] epoch {ep+1}/{train_epochs} - train_loss={train_loss_sum/len(train_loader):.5f} val_loss={val_loss_sum/len(val_loader):.5f}")

    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim, "player_names": ["A", "B", "C", "D"]}, model_path)
    print(f"[train] 模型已保存到 {model_path}")
    return NeuralAssistWrapper(model, ["A", "B", "C", "D"], device="cpu")

def resume_train_assist_model(model_path="pairwise_assist_cpu.pt", extra_games=1000, extra_epochs=3, batch_size=1024, lr=5e-4):
    if not os.path.exists(model_path):
        return train_cpu_assist_model(model_path, num_games=extra_games, train_epochs=extra_epochs, batch_size=batch_size, lr=lr)

    ckpt = torch.load(model_path, map_location="cpu")
    input_dim = ckpt["input_dim"]
    model = PairwiseAssistNet(input_dim=input_dim, output_dim=4)
    model.load_state_dict(ckpt["state_dict"])

    print(f"[assist] 成功挂载旧模型 {model_path}，开始续训进阶...")
    print("[train] 开始采样无噪声新增监督数据...")
    samples = collect_supervised_samples(num_games=extra_games, max_small_rounds=4)

    xs = torch.tensor([s[0] for s in samples], dtype=torch.float32)
    ys = torch.tensor([s[1] for s in samples], dtype=torch.float32)
    ms = torch.tensor([s[2] for s in samples], dtype=torch.float32)

    n = xs.shape[0]
    perm = torch.randperm(n)
    split = int(n * 0.9)

    train_ds = TensorDataset(xs[perm[:split]], ys[perm[:split]], ms[perm[:split]])
    val_ds = TensorDataset(xs[perm[split:]], ys[perm[split:]], ms[perm[split:]])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(extra_epochs):
        model.train()
        train_loss_sum = 0.0
        for xb, yb, mb in train_loader:
            optimizer.zero_grad()
            loss = masked_bce_with_logits(model(xb), yb, mb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                val_loss_sum += masked_bce_with_logits(model(xb), yb, mb).item()

        print(f"[train] continue epoch {ep+1}/{extra_epochs} - train_loss={train_loss_sum/len(train_loader):.5f} val_loss={val_loss_sum/len(val_loader):.5f}")

    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim, "player_names": ckpt["player_names"]}, model_path)
    return NeuralAssistWrapper(model, ckpt["player_names"], device="cpu")

def ensure_assist_model(model_path="pairwise_assist_cpu.pt", continue_train=False, extra_games=1000, extra_epochs=3):
    if not TORCH_AVAILABLE: return None
    if not os.path.exists(model_path):
        return train_cpu_assist_model(model_path=model_path, num_games=1500, train_epochs=5, batch_size=1024, lr=1e-3)
    if continue_train:
        return resume_train_assist_model(model_path=model_path, extra_games=extra_games, extra_epochs=extra_epochs, batch_size=1024, lr=5e-4)
    
    ckpt = torch.load(model_path, map_location="cpu")
    model = PairwiseAssistNet(input_dim=ckpt["input_dim"], output_dim=4)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[assist] 已加载现有模型: {model_path}")
    return NeuralAssistWrapper(model, ckpt["player_names"], device="cpu")


_GLOBAL_ASSIST_MODEL = None
_GLOBAL_MODEL_PATH = "pairwise_assist_cpu.pt"

CONTINUE_TRAIN = True       
EXTRA_TRAIN_GAMES = 100    
EXTRA_TRAIN_EPOCHS = 3       

def get_global_assist_model():
    global _GLOBAL_ASSIST_MODEL
    if _GLOBAL_ASSIST_MODEL is None:
        _GLOBAL_ASSIST_MODEL = ensure_assist_model(
            model_path=_GLOBAL_MODEL_PATH,
            continue_train=CONTINUE_TRAIN,
            extra_games=EXTRA_TRAIN_GAMES,
            extra_epochs=EXTRA_TRAIN_EPOCHS
        )
    return _GLOBAL_ASSIST_MODEL

# =========================
# 批量生成及导出
# =========================

def generate_one_game(num_games):
    players = ["A", "B", "C", "D"]
    game = NarrativePokerGame(
        player_names=players,
        seed=None,
        max_small_rounds=4,
        assist_model=get_global_assist_model(),
        collect_training=False
    )
    game.play_game()
    report = game.render_report(num_games)
    reward = game.get_reward()
    jsonl_dict = game.to_jsonl_dict()
    return report, reward, jsonl_dict

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 启动【高级量化博弈引擎】(附带 JSONL 思维链导出) 🚀")
    print("= 已彻底修复：无限死锁 & 0战力空气牌扎堆撞车缺陷 =")
    
    texts = []
    jsonl_records = []
    epochs = 200
    correct = 0

    for i in range(epochs):
        text, reward, jsonl_dict = generate_one_game(i + 1)
        texts.append(text)
        if jsonl_dict:
            jsonl_records.append(jsonl_dict)
        if reward == 1.0:
            correct += 1

    acc = correct / epochs if epochs > 0 else 0.0

    summary = []
    summary.append(f"总对局数: {epochs}")
    summary.append(f"命中局数: {correct}")
    summary.append(f"预测准确率: {acc:.2%}")
    summary_text = "\n".join(summary)

    print(summary_text)

    # 1. 导出战报
    if os.path.exists("mock_game.txt"):
        os.remove("mock_game.txt")

    with open("mock_game.txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n\n")
        for text in texts:
            f.write(text + "\n\n")
            
    # 2. 导出模型思维链训练数据 (严格按要求转化 1-indexed)
    jsonl_filename = "poker_training_data.jsonl"
    with open(jsonl_filename, "w", encoding="utf-8") as f:
        for record in jsonl_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"\n✅ 执行完毕！")
    print(f"- 文本报表已保存: mock_game.txt")
    print(f"- 模型大本营已导出: {jsonl_filename} (共计 {len(jsonl_records)} 条)")