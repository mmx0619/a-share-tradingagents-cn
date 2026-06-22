"""A 股 Portfolio Manager。

这个文件对应原版 TradingAgents 的：

    tradingagents/agents/managers/portfolio_manager.py

Portfolio Manager 是当前交易研究链路的最终裁判。

它读取：

    1. Research Manager 的 investment_plan；
    2. Trader Agent 的 trader_plan；
    3. 三位 Risk Analyst 的风险辩论历史；

然后输出最终交易决策。

最终评级仍然使用原版的五档：

    Buy
    Overweight
    Hold
    Underweight
    Sell

注意：
    Trader Agent 输出的是 Buy / Hold / Sell 三档交易动作。
    Portfolio Manager 输出的是更完整的组合评级，
    可以表达“低配 / 超配”这类比单次交易更细的仓位态度。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from tradingagents_cn.agents.research_manager import PortfolioRating
from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


class PortfolioDecision(BaseModel):
    """Portfolio Manager 输出的结构化最终交易决策。

    rating:
        最终组合评级，必须是 Buy / Overweight / Hold / Underweight / Sell。

    executive_summary:
        简短执行摘要。

    investment_thesis:
        具体投资逻辑，必须基于研究计划、交易提案和风险辩论。

    price_target:
        可选目标价。
        如果材料不足以支持明确价格，返回 null。

    time_horizon:
        可选持有周期，例如“1-4 周”、“3-6 个月”。
    """

    rating: PortfolioRating = Field(
        description="Exactly one of Buy / Overweight / Hold / Underweight / Sell.",
    )
    executive_summary: str = Field(description="最终执行摘要。")
    investment_thesis: str = Field(description="基于证据的最终投资逻辑。")
    price_target: Optional[float] = Field(default=None, description="可选目标价。")
    time_horizon: Optional[str] = Field(default=None, description="可选持有周期。")


PORTFOLIO_MANAGER_SYSTEM_PROMPT = """你是 A 股 Portfolio Manager，负责输出最终交易决策。

你会收到：

1. Research Manager 的 investment_plan。
2. Trader Agent 的 trader_plan。
3. Aggressive / Conservative / Neutral 三位风险分析员的风险辩论历史。

你的任务：

1. 综合研究计划、交易提案和风险辩论。
2. 选择一个最终组合评级：
   Buy / Overweight / Hold / Underweight / Sell。
3. 给出执行摘要和投资逻辑。
4. 如果材料支持，可以给出 price_target 和 time_horizon。
   如果材料不足，price_target 返回 null。

评级解释：

- Buy:
  强烈建议进入或增加仓位。

- Overweight:
  看法偏积极，可以逐步提高仓位。

- Hold:
  保持当前状态，暂不主动加减仓。

- Underweight:
  降低风险敞口，减仓或低配。

- Sell:
  退出或回避。

输出要求：

你必须只输出一个合法 JSON 对象，不要输出 Markdown，不要加解释文字。

JSON 格式必须是：

{
  "rating": "Buy | Overweight | Hold | Underweight | Sell",
  "executive_summary": "最终执行摘要",
  "investment_thesis": "最终投资逻辑",
  "price_target": 123.45 或 null,
  "time_horizon": "持有周期或 null"
}

本最终决策用于个人投资研究辅助，最终决策由使用者自行确认。
"""


@dataclass
class PortfolioManagerContext:
    """Portfolio Manager 的输入上下文。"""

    symbol: str
    trade_date: str
    prompt: str


def build_portfolio_manager_context(
    symbol: str,
    trade_date: str,
    investment_plan: str,
    trader_plan: str,
    risk_debate_history: str,
    past_context: str = "",
) -> PortfolioManagerContext:
    """构造 Portfolio Manager 上下文。"""
    normalized_symbol = normalize_cn_symbol(symbol)
    prompt = build_portfolio_manager_prompt(
        symbol=normalized_symbol,
        trade_date=trade_date,
        investment_plan=investment_plan,
        trader_plan=trader_plan,
        risk_debate_history=risk_debate_history,
        past_context=past_context,
    )
    return PortfolioManagerContext(
        symbol=normalized_symbol,
        trade_date=trade_date,
        prompt=prompt,
    )


def build_portfolio_manager_prompt(
    symbol: str,
    trade_date: str,
    investment_plan: str,
    trader_plan: str,
    risk_debate_history: str,
    past_context: str = "",
) -> str:
    """构造 Portfolio Manager Prompt。"""
    past_context_section = ""
    if past_context:
        past_context_section = f"""
下面是历史交易记忆和复盘经验：

{past_context}
"""

    return f"""{PORTFOLIO_MANAGER_SYSTEM_PROMPT}

当前分析股票：{symbol}
当前分析日期：{trade_date}

下面是 Research Manager 的 investment_plan：

{investment_plan}

下面是 Trader Agent 的 trader_plan：

{trader_plan}

下面是三位风险分析员的风险辩论历史：

{risk_debate_history}
{past_context_section}

请基于以上内容输出 PortfolioDecision JSON。
"""


def parse_portfolio_decision_from_text(text: str) -> PortfolioDecision:
    """从模型文本中解析并校验 PortfolioDecision。"""
    json_text = extract_json_object_text(text)
    data = json.loads(json_text)
    return PortfolioDecision.model_validate(data)


def extract_json_object_text(text: str) -> str:
    """从文本中提取 JSON 对象字符串。"""
    stripped = text.strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S)
    if fenced_match:
        return fenced_match.group(1)

    object_match = re.search(r"\{.*\}", stripped, re.S)
    if object_match:
        return object_match.group(0)

    raise ValueError("模型输出中没有找到 JSON 对象。")


def render_portfolio_decision(decision: PortfolioDecision) -> str:
    """把结构化 PortfolioDecision 渲染成 Markdown。"""
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]

    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])

    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])

    return "\n".join(parts)


def portfolio_decision_to_json_schema() -> dict:
    """返回 PortfolioDecision 的 JSON Schema。"""
    return PortfolioDecision.model_json_schema()


def build_fallback_portfolio_decision(error_message: str) -> PortfolioDecision:
    """构造 Portfolio Manager 的保守兜底输出。"""
    return PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="模型没有返回合法最终决策，系统进入保守持有/观望兜底。",
        investment_thesis=(
            "由于 Portfolio Manager 输出未通过结构化校验，"
            "本次不应基于非结构化文本做激进交易。"
            f"错误：{error_message}"
        ),
        price_target=None,
        time_horizon="等待重新分析",
    )
