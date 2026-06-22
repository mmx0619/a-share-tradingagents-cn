"""A 股情绪面 Analyst。

Sentiment Analyst 对应原项目里的社媒/情绪分析角色。

在美股原项目里，常见原材料是 Reddit、StockTwits 等社区讨论。
在 A 股版里，对应的公开来源是：

    东方财富股吧、雪球、同花顺股吧、淘股吧等。

注意：
    情绪面不是“大家看涨就一定涨”。
    它只是观察市场参与者的注意力、分歧、恐慌、过热和题材传播情况。
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


SENTIMENT_ANALYST_SYSTEM_PROMPT = """你是 A 股情绪面分析员 Sentiment Analyst。

你的任务是基于公开社区讨论材料，分析市场情绪和投资者关注点。

你重点观察：

1. 讨论热度：是否出现明显关注度上升或异常活跃。
2. 情绪方向：讨论更偏乐观、悲观、分歧，还是噪声较多。
3. 题材传播：是否有反复出现的主题、概念、事件或预期。
4. 风险信号：是否出现过度一致、恐慌扩散、谣言化表达、无证据狂热。
5. 证据质量：材料是否只是标题，是否缺少正文，是否来源单一或数量不足。

输出要求：

1. 必须明确说明材料来源和质量。
2. 不要把社区情绪直接等同于股价涨跌。
3. 不要编造没有出现在工具结果里的帖子、观点或数据。
4. 不要直接给最终买入/卖出结论。
5. 使用 Markdown 输出。
6. 最后用 Markdown 表格整理：
   - 情绪维度；
   - 当前观察；
   - 证据强度；
   - 对后续交易决策的意义；
   - 需要继续核实的问题。
"""


@dataclass
class SentimentAgentContext:
    """Sentiment Agent 的输入上下文。"""

    symbol: str
    trade_date: str
    sentiment_materials: str
    prompt: str


def build_sentiment_agent_context(
    symbol: str,
    trade_date: str,
    sentiment_materials: str,
) -> SentimentAgentContext:
    """构造 Sentiment Agent 上下文。

    这个函数适合普通 Prompt 流程使用。
    Tool Calling 版则会让模型先调用 get_stock_sentiment 工具，再生成报告。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    prompt = build_sentiment_agent_prompt(
        symbol=normalized_symbol,
        trade_date=trade_date,
        sentiment_materials=sentiment_materials,
    )
    return SentimentAgentContext(
        symbol=normalized_symbol,
        trade_date=trade_date,
        sentiment_materials=sentiment_materials,
        prompt=prompt,
    )


def build_sentiment_agent_prompt(
    symbol: str,
    trade_date: str,
    sentiment_materials: str,
) -> str:
    """构造给大模型的 Sentiment Agent Prompt。"""
    return f"""{SENTIMENT_ANALYST_SYSTEM_PROMPT}

当前分析股票：{symbol}
当前分析日期：{trade_date}

下面是情绪面原材料：

{sentiment_materials}

请基于以上材料，撰写 A 股情绪面分析报告。
"""


def render_sentiment_agent_context(context: SentimentAgentContext) -> str:
    """渲染 Sentiment Agent 上下文，方便调试时阅读。"""
    return context.prompt
