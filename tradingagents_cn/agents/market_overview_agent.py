"""A 股市场概览 Agent。

这个 Agent 服务于用户的问题：

    今天股市怎么样？
    大盘现在什么情况？

它读取市场概览原材料：
    - 主要指数；
    - 涨跌家数；
    - 行业板块强弱；

然后生成一份人能直接阅读的大盘分析报告。
"""

from __future__ import annotations

from dataclasses import dataclass


MARKET_OVERVIEW_SYSTEM_PROMPT = """你是 A 股市场概览分析师。

你的任务是根据给定的市场原材料，写一份简洁但有判断力的大盘分析报告。

分析要求：
1. 先给出一句明确的市场状态判断，例如：偏强、震荡、偏弱、分化明显。
2. 分析主要指数表现，不要编造原材料中没有的数字。
3. 分析涨跌家数，判断市场宽度。
4. 分析行业板块强弱，指出领涨和偏弱方向。
5. 最后给出短线观察重点。
6. 不要推荐具体股票。
7. 不要写成投资鸡汤。
8. 使用中文 Markdown。
"""


@dataclass
class MarketOverviewAgentContext:
    """市场概览 Agent 上下文。"""

    question: str
    materials: str
    prompt: str


def build_market_overview_agent_context(
    question: str,
    materials: str,
) -> MarketOverviewAgentContext:
    """构造市场概览 Agent 上下文。"""
    prompt = build_market_overview_agent_prompt(
        question=question,
        materials=materials,
    )
    return MarketOverviewAgentContext(
        question=question,
        materials=materials,
        prompt=prompt,
    )


def build_market_overview_agent_prompt(question: str, materials: str) -> str:
    """构造市场概览 Agent Prompt。"""
    return f"""{MARKET_OVERVIEW_SYSTEM_PROMPT}

用户问题：
{question}

下面是市场概览原材料：

{materials}

请基于以上原材料生成 A 股市场概览报告。
"""


def render_market_overview_agent_context(context: MarketOverviewAgentContext) -> str:
    """渲染市场概览 Agent 上下文，方便调试。"""
    return context.prompt
