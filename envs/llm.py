"""
最小化的LLM交互模块
"""

from pydoc import text

from openai import OpenAI


class LLMAgent:
    """LLM代理，封装模型调用逻辑"""
    
    def __init__(self, base_url="http://localhost:8000/v1", model_name="Qwen3-1.7B"):
        """初始化LLM客户端"""
        self.client = OpenAI(base_url=base_url, api_key="no-api-key-required")
        self.model_name = model_name
    
    def get_action(self, player_info):
        """
        获取LLM的动作决策
        
        Args:
            player_info: dict, 包含玩家状态信息
        
        Returns:
            int: 目标位置 (0到n-1之间的整数)
        """
        try:
            # 构建提示词
            prompt = self._build_prompt(player_info)
            
            # 调用模型
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.7,
            )
            # print(f"LLM响应: {response.choices[0].message.content.strip()}")
            
            # 解析响应
            output = response.choices[0].message.content.strip()
            
            # 提取数字
            result = self._extract_position(output, player_info["max_position"])
            print(f"LLM决策: {result}")
            return result
            
        except Exception:
            # 失败时返回当前位置（不行动）
            print("LLM请求失败，返回当前位置")
            return player_info["current_pos"]
    
    def _build_prompt(self, info):
        """构建提示词"""
        return """
# 任务说明：德州扑克位次博弈 (Poker Positioning Game)

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
* 这个数字代表你想要移动到的**目标索引 (aim)**，范围是 $1$ 到 $N$。
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
3.  **达成共识**：如果当前位次已经正确反映了你的实力范围（包括平局情况），你应该选择保持当前位置，以尽快结束回合获取奖励。"""+f"""
current_order：{info['current_order_str']},history：{info['history']}
current_pos：{info['current_pos']}，hand：{info['hand']}
community：{info['community']}，stage：{info['stage']}
"""
    
    def _extract_position(self, text, max_pos):
        import re
        # 匹配\boxed{}内的内容
        boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', text)
        if boxed_matches:
            boxed_text = boxed_matches[0]  # 取第一个\boxed{}内的文本
            numbers = re.findall(r'\d+', boxed_text)
            if numbers:
                num = int(numbers[0])
                # 确保在有效范围内
                if 1 <= num <= max_pos:
                    return num
        # 默认返回中间位置
        return max_pos // 2


# 全局实例
_agent = None

def get_agent():
    """获取全局LLM代理实例"""
    global _agent
    if _agent is None:
        _agent = LLMAgent()
    return _agent

def request_action(player, stage, current_pos, player_hand, 
                   community_cards, current_order, history_orders):
    """
    请求LLM动作的接口函数
    
    Args:
        player: 玩家名称
        stage: 游戏阶段
        current_pos: 当前位置
        player_hand: 玩家手牌列表
        community_cards: 公共牌列表
        current_order: 当前位次列表
        history_orders: 历史位次记录
    
    Returns:
        int: 目标位置
    """
    # 准备玩家信息
    player_info = {
        "player": player,
        "stage": stage,
        "current_pos": current_pos,
        "hand": " ".join(str(card) for card in player_hand),
        "community": " ".join(str(card) for card in community_cards),
        "current_order_str": " -> ".join(current_order),
        "max_position": len(current_order),
        "history": history_orders
    }
    
    # 获取代理并请求动作
    # print(f"请求LLM动作，玩家信息: {player_info}")
    agent = get_agent()
    return agent.get_action(player_info)
