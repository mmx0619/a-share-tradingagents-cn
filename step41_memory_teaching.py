"""第 41 步：看懂 Memory 记忆。

Checkpoint 和 Memory 很容易混在一起。

它们不一样：

Checkpoint：
    保存“当前这一次流程跑到哪里了”。

Memory：
    保存“过去发生过什么、学到了什么”，下次可以拿出来参考。

举例：

Checkpoint 像游戏存档：
    我当前打到第几关、血量多少、背包有什么。

Memory 像经验笔记：
    上次遇到类似股票，龙虎榜高换手后第二天继续大跌。
    下次再遇到类似情况，要提醒风控谨慎。

在 A 股多 Agent 项目里，Memory 可以保存：

1. 历史分析报告。
2. 盘后复盘结论。
3. 某只股票过去的风险事件。
4. 某类题材或形态的经验。
5. 模型上次判断错在哪里。

本文件做一个最小教学版：

第一次：
    保存一条 002361 的复盘记忆。

第二次：
    用户再次分析 002361。
    程序先从 Memory 中取出相关记忆。
    再把记忆放进 Prompt，帮助 Agent 分析。

为了学习清楚：

- 不联网。
- 不调用真实大模型。
- 不导入前面复杂模块。
- 用普通 list 模拟记忆库。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryItem:
    """一条记忆。

    字段说明：

    - symbol：股票代码。
    - memory_type：记忆类型，例如 review、risk、lesson。
    - content：记忆内容。
    """

    symbol: str
    memory_type: str
    content: str


class SimpleMemoryStore:
    """一个最简单的内存记忆库。

    真实项目里，Memory 不应该只放在 Python list 里。

    真实项目可以放到：

    - SQLite
    - PostgreSQL
    - Redis
    - 向量数据库
    - 本地 JSON 文件

    这里为了教学，只用 list。
    """

    def __init__(self) -> None:
        """初始化空记忆库。"""
        self.items: list[MemoryItem] = []

    def add_memory(self, item: MemoryItem) -> None:
        """保存一条记忆。"""
        self.items.append(item)

    def search_by_symbol(self, symbol: str) -> list[MemoryItem]:
        """按股票代码检索记忆。"""
        return [
            item
            for item in self.items
            if item.symbol == symbol
        ]


def render_memories(memories: list[MemoryItem]) -> str:
    """把记忆列表渲染成 Prompt 可读的文本。"""
    if not memories:
        return "暂无历史记忆。"

    lines: list[str] = []
    for index, item in enumerate(memories, start=1):
        lines.append(f"记忆 {index}")
        lines.append(f"股票代码：{item.symbol}")
        lines.append(f"记忆类型：{item.memory_type}")
        lines.append(f"内容：{item.content}")
        lines.append("")

    return "\n".join(lines)


def build_analysis_prompt(
    symbol: str,
    current_market_text: str,
    memory_text: str,
) -> str:
    """构造带历史记忆的分析 Prompt。

    重点看：

    当前行情信息
      +
    历史记忆
      ↓
    一起放进 Prompt

    这样 Agent 分析时就不是只看当前数据，
    还可以参考历史复盘经验。
    """
    return f"""你是 A 股多智能体投研系统里的研究 Agent。

请结合当前信息和历史记忆做分析。

注意：
1. 历史记忆只能作为参考，不能机械套用。
2. 如果当前情况和历史记忆相似，要说明相似点。
3. 如果当前情况已经不同，也要说明差异。
4. 不要直接给确定性买卖指令。

股票代码：
{symbol}

当前行情信息：
{current_market_text}

历史记忆：
{memory_text}
"""


def mock_agent_with_memory(prompt: str) -> str:
    """模拟读取 Memory 后的大模型回答。

    真实项目里，这里会调用 DeepSeek 或其他模型。

    教学版只返回固定文本。
    """
    if "龙虎榜高换手后次日继续走弱" in prompt:
        return (
            "记忆增强分析：当前仍有高换手和短线博弈特征，"
            "历史记忆显示类似情况下曾出现次日继续走弱。"
            "因此本次应提醒风控维持谨慎，不应仅因小幅反弹就下调风险。"
        )

    return "普通分析：暂无可用历史记忆，只基于当前信息分析。"


def run_memory_demo() -> str:
    """运行 Memory 教学演示。"""
    memory_store = SimpleMemoryStore()

    # 第一次：保存盘后复盘记忆。
    #
    # 这相当于系统在一次分析结束后，
    # 把有价值的经验写入长期记忆。
    memory_store.add_memory(
        MemoryItem(
            symbol="002361",
            memory_type="review",
            content=(
                "龙虎榜高换手后次日继续走弱。"
                "复盘结论：短线资金博弈强时，小幅反弹不一定代表风险解除。"
            ),
        )
    )

    # 第二次：再次分析同一只股票时，先取出历史记忆。
    memories = memory_store.search_by_symbol("002361")
    memory_text = render_memories(memories)

    current_market_text = (
        "实时行情显示 002361 小幅反弹，"
        "但近期仍存在高换手、龙虎榜和短线剧烈波动。"
    )
    prompt = build_analysis_prompt(
        symbol="002361",
        current_market_text=current_market_text,
        memory_text=memory_text,
    )
    report = mock_agent_with_memory(prompt)

    return f"""======== 记忆库中保存的内容 ========
{memory_text}

======== 带 Memory 的 Prompt ========
{prompt}

======== Agent 读取 Memory 后的输出 ========
{report}
"""


if __name__ == "__main__":
    print(run_memory_demo())
