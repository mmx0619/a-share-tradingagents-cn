"""A 股选股/筛股 Agent。

这个 Agent 读取候选股票池原材料，
生成一份“值得继续研究的关注名单”。

注意：
    它不是最终 Portfolio Manager。
    它不能直接给出确定买入指令。
    对单只股票是否能买，仍然应该进入单股完整 TradingAgents 链路。
"""

from __future__ import annotations

from dataclasses import dataclass


STOCK_SCREENING_SYSTEM_PROMPT = """你是 A 股选股研究员。

你会收到一份候选股票池原材料。

你的任务：
1. 从候选里挑出最值得继续研究的股票，最多 5 只。
2. 给出每只股票入选的具体原因，例如涨幅、成交额、换手率、量比等。
3. 如果原材料里有 Sector、DynamicPE、PB、TotalMarketCap 等字段，可以作为辅助筛选理由。
4. 明确说明这些估值和市值字段只是轻量过滤，不等于完整基本面结论。
5. 明确说明这只是候选观察名单，不是直接买入建议。
6. 提醒下一步应该对候选股分别跑单股完整分析。
7. 不要编造原材料里没有的数据。
8. 使用中文 Markdown。
"""


@dataclass
class StockScreeningAgentContext:
    """选股 Agent 上下文。"""

    question: str
    materials: str
    prompt: str


def build_stock_screening_agent_context(
    question: str,
    materials: str,
) -> StockScreeningAgentContext:
    """构造选股 Agent 上下文。"""
    prompt = build_stock_screening_agent_prompt(
        question=question,
        materials=materials,
    )
    return StockScreeningAgentContext(
        question=question,
        materials=materials,
        prompt=prompt,
    )


def build_stock_screening_agent_prompt(question: str, materials: str) -> str:
    """构造选股 Agent Prompt。"""
    return f"""{STOCK_SCREENING_SYSTEM_PROMPT}

用户问题：
{question}

下面是候选股票池原材料：

{materials}

请基于以上原材料生成 A 股候选观察名单。
"""


def render_stock_screening_agent_context(context: StockScreeningAgentContext) -> str:
    """渲染选股 Agent 上下文，方便调试。"""
    return context.prompt
