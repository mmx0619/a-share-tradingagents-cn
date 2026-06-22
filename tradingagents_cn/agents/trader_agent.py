"""A 股 Trader Agent。

这个文件对应原版 TradingAgents 里的：

    tradingagents/agents/trader/trader.py

原版 Trader 的作用是：

    读取 Research Manager 的 investment_plan，
    然后把它转换成更具体的交易提案。

Research Manager 输出的是五档评级：

    Buy / Overweight / Hold / Underweight / Sell

Trader 输出的是更直接的三档交易动作：

    Buy / Hold / Sell

为什么这里要再过一层 Trader？
    因为 Research Manager 更像“研究负责人”，
    它给出方向和策略；
    Trader 更像“执行计划制定者”，
    它要给出动作、理由、入场价格、止损、仓位。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


class TraderAction(str, Enum):
    """Trader 使用的三档交易动作。"""

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


class TraderProposal(BaseModel):
    """Trader 输出的结构化交易提案。

    action:
        交易动作，必须是 Buy / Hold / Sell。

    reasoning:
        做出这个动作的理由。

    entry_price:
        可选入场价格。
        如果当前不适合给明确价格，可以返回 null。

    stop_loss:
        可选止损价格。
        如果当前不适合给明确价格，可以返回 null。

    position_sizing:
        可选仓位建议，例如“总资金 5% 以内”。
    """

    action: TraderAction = Field(description="Exactly one of Buy / Hold / Sell.")
    reasoning: str = Field(description="基于研究计划和报告，解释交易动作。")
    entry_price: Optional[float] = Field(default=None, description="可选入场价格。")
    stop_loss: Optional[float] = Field(default=None, description="可选止损价格。")
    position_sizing: Optional[str] = Field(default=None, description="可选仓位建议。")


TRADER_SYSTEM_PROMPT = """你是 A 股 Trader Agent，任务是把研究计划转换成具体交易提案。

你会收到：

1. Market Agent 技术面报告。
2. News Agent 新闻面报告。
3. Fundamentals Agent 基本面报告。
4. Research Manager 给出的 investment_plan。

你的任务：

1. 输出一个明确交易动作：
   Buy / Hold / Sell 三选一。

2. 给出简短但具体的 reasoning。

3. 如果材料支持，可以给出 entry_price 和 stop_loss。
   如果材料不足以支持明确价格，请返回 null，不要编造价格。

4. 给出 position_sizing 仓位建议。
   仓位建议要和 Research Manager 的 recommendation 一致。

动作映射参考：

- Research Manager = Buy:
  Trader 通常可以考虑 Buy。

- Research Manager = Overweight:
  Trader 可以考虑 Buy 或 Hold，取决于是否已有明确入场条件。

- Research Manager = Hold:
  Trader 通常应输出 Hold。

- Research Manager = Underweight:
  Trader 可以考虑 Hold 或 Sell，取决于风险强度。

- Research Manager = Sell:
  Trader 通常可以考虑 Sell。

输出要求：

你必须只输出一个合法 JSON 对象，不要输出 Markdown，不要加解释文字。

JSON 格式必须是：

{
  "action": "Buy | Hold | Sell",
  "reasoning": "做出该交易动作的理由",
  "entry_price": 123.45 或 null,
  "stop_loss": 123.45 或 null,
  "position_sizing": "仓位建议"
}

本交易提案用于个人投资研究辅助，最终决策由使用者自行确认。
"""


@dataclass
class TraderAgentContext:
    """Trader Agent 的输入上下文。"""

    symbol: str
    trade_date: str
    prompt: str


def build_trader_agent_context(
    symbol: str,
    trade_date: str,
    market_report: str,
    news_report: str,
    fundamentals_report: str,
    investment_plan: str,
) -> TraderAgentContext:
    """构造 Trader Agent 上下文。"""
    normalized_symbol = normalize_cn_symbol(symbol)
    prompt = build_trader_agent_prompt(
        symbol=normalized_symbol,
        trade_date=trade_date,
        market_report=market_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        investment_plan=investment_plan,
    )
    return TraderAgentContext(
        symbol=normalized_symbol,
        trade_date=trade_date,
        prompt=prompt,
    )


def build_trader_agent_prompt(
    symbol: str,
    trade_date: str,
    market_report: str,
    news_report: str,
    fundamentals_report: str,
    investment_plan: str,
) -> str:
    """构造 Trader Agent Prompt。"""
    return f"""{TRADER_SYSTEM_PROMPT}

当前分析股票：{symbol}
当前分析日期：{trade_date}

下面是 Market Agent 技术面报告：

{market_report}

下面是 News Agent 新闻面报告：

{news_report}

下面是 Fundamentals Agent 基本面报告：

{fundamentals_report}

下面是 Research Manager 给出的 investment_plan：

{investment_plan}

请基于以上内容输出 TraderProposal JSON。
"""


def parse_trader_proposal_from_text(text: str) -> TraderProposal:
    """从模型文本中解析并校验 TraderProposal。"""
    json_text = extract_json_object_text(text)
    data = json.loads(json_text)
    return TraderProposal.model_validate(data)


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


def render_trader_proposal(proposal: TraderProposal) -> str:
    """把结构化 TraderProposal 渲染成 Markdown。"""
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]

    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])

    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])

    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])

    parts.extend(["", f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**"])
    return "\n".join(parts)


def trader_proposal_to_json_schema() -> dict:
    """返回 TraderProposal 的 JSON Schema。"""
    return TraderProposal.model_json_schema()


def build_fallback_trader_proposal(error_message: str) -> TraderProposal:
    """构造 Trader Agent 的保守兜底输出。"""
    return TraderProposal(
        action=TraderAction.HOLD,
        reasoning=(
            "模型没有返回可校验的 TraderProposal，进入保守兜底。"
            f"错误：{error_message}"
        ),
        entry_price=None,
        stop_loss=None,
        position_sizing="暂不新增仓位，等待结构化交易提案恢复后再评估。",
    )
