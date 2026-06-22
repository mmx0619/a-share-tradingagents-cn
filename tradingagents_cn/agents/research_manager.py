"""A 股 Research Manager。

这个文件对应原版 TradingAgents 里的：

    tradingagents/agents/managers/research_manager.py

原版 Research Manager 的作用是：

    读取 Bull / Bear 的辩论历史，
    然后给 Trader 一个清晰的 investment plan。

它会从固定评级里选一个：

    Buy
    Overweight
    Hold
    Underweight
    Sell

当前 A 股版也保留这个结构。

为什么要用固定评级？
    因为后面的 Trader / Risk / Portfolio Manager 需要稳定字段。
    如果模型一会儿输出“积极看多”，一会儿输出“可以考虑”，
    程序就很难继续自动流转。

所以这里用 Pydantic + Enum 做校验：

    模型必须返回合法 JSON；
    recommendation 必须是五个固定值之一；
    否则程序会报错，后续再加重试或保守兜底。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


class PortfolioRating(str, Enum):
    """Research Manager 使用的 5 档评级。

    这里保留英文值，是为了贴近原版 TradingAgents。
    中文解释放在 Prompt 里。
    """

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class ResearchPlan(BaseModel):
    """Research Manager 输出的结构化研究计划。

    recommendation:
        五档评级，必须严格是：
            Buy / Overweight / Hold / Underweight / Sell

    rationale:
        为什么选这个评级。

    strategic_actions:
        给后续 Trader 的具体执行建议。
        例如仓位、观察条件、触发条件、失效条件。
    """

    recommendation: PortfolioRating = Field(
        description="Exactly one of Buy / Overweight / Hold / Underweight / Sell.",
    )
    rationale: str = Field(
        description="结合多头和空头辩论，解释为什么这个评级更合理。",
    )
    strategic_actions: str = Field(
        description="给后续 Trader 的具体执行建议，包括仓位、触发条件、失效条件等。",
    )


RESEARCH_MANAGER_SYSTEM_PROMPT = """你是 A 股 Research Manager，也是多空辩论的裁判。

你的任务：

1. 阅读 Bull Researcher 和 Bear Researcher 的辩论历史。
2. 判断哪一方的证据更强。
3. 输出一个清晰、可执行的个人投资研究计划。

评级只能从下面五个里选一个：

- Buy:
  强烈偏多。多头证据明显强于空头证据，可以考虑建立或增加仓位。

- Overweight:
  偏多但不是极端确定。可以考虑逐步增加风险敞口。

- Hold:
  多空证据相对均衡，或者关键信息不足。维持观察或保持现有仓位。

- Underweight:
  偏谨慎。空头风险较突出，可以考虑降低风险敞口。

- Sell:
  强烈偏空。空头证据明显强于多头证据，可以考虑回避或退出。

不要轻易使用 Hold。
只有当双方证据确实均衡，或者关键信息明显不足时，才使用 Hold。

输出要求：

你必须只输出一个合法 JSON 对象，不要输出 Markdown，不要加解释文字。

JSON 格式必须是：

{
  "recommendation": "Buy | Overweight | Hold | Underweight | Sell",
  "rationale": "说明为什么选择这个评级",
  "strategic_actions": "给后续 Trader 的具体执行建议"
}

本计划用于个人投资研究辅助，最终决策由使用者自行确认。
"""


@dataclass
class ResearchManagerContext:
    """Research Manager 的输入上下文。"""

    symbol: str
    trade_date: str
    debate_history: str
    prompt: str


def build_research_manager_context(
    symbol: str,
    trade_date: str,
    debate_history: str,
) -> ResearchManagerContext:
    """构造 Research Manager 上下文。"""
    normalized_symbol = normalize_cn_symbol(symbol)
    prompt = build_research_manager_prompt(
        symbol=normalized_symbol,
        trade_date=trade_date,
        debate_history=debate_history,
    )
    return ResearchManagerContext(
        symbol=normalized_symbol,
        trade_date=trade_date,
        debate_history=debate_history,
        prompt=prompt,
    )


def build_research_manager_prompt(
    symbol: str,
    trade_date: str,
    debate_history: str,
) -> str:
    """构造 Research Manager Prompt。"""
    return f"""{RESEARCH_MANAGER_SYSTEM_PROMPT}

当前分析股票：{symbol}
当前分析日期：{trade_date}

下面是多空辩论历史：

{debate_history}

请基于以上辩论历史，输出 ResearchPlan JSON。
"""


def render_research_plan(plan: ResearchPlan) -> str:
    """把结构化 ResearchPlan 渲染成 Markdown。

    后续 Trader Agent 更适合读取 Markdown 文本，
    所以这里把 JSON 对象变成人类也容易读的格式。
    """
    return "\n".join(
        [
            f"**Recommendation**: {plan.recommendation.value}",
            "",
            f"**Rationale**: {plan.rationale}",
            "",
            f"**Strategic Actions**: {plan.strategic_actions}",
        ]
    )


def parse_research_plan_from_text(text: str) -> ResearchPlan:
    """从模型文本中解析并校验 ResearchPlan。

    理想情况下，模型会直接返回 JSON。
    但有些模型可能包一层 ```json 代码块。

    这个函数做两步：
        1. 尽量提取 JSON；
        2. 用 Pydantic 校验字段和值。

    如果 recommendation 不是固定五档之一，
    这里会直接报错。
    """
    json_text = extract_json_object_text(text)
    data = json.loads(json_text)
    return ResearchPlan.model_validate(data)


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


def research_plan_to_json_schema() -> dict[str, Any]:
    """返回 ResearchPlan 的 JSON Schema。

    当前 DeepSeek 调用先用 Prompt 约束 JSON。
    这个 schema 先暴露出来，后续接 OpenAI structured output
    或其他支持 response_format 的模型时可以直接复用。
    """
    return ResearchPlan.model_json_schema()


def build_fallback_research_plan(error_message: str) -> ResearchPlan:
    """构造 Research Manager 的保守兜底输出。

    如果模型多次没有返回合法 JSON，程序不能继续拿普通文本往下走。
    所以这里返回合法 ResearchPlan，评级使用 Hold。
    """
    return ResearchPlan(
        recommendation=PortfolioRating.HOLD,
        rationale=(
            "模型没有返回可校验的 ResearchPlan，进入保守兜底。"
            f"错误：{error_message}"
        ),
        strategic_actions="暂停主动加仓，等待结构化研究结论恢复后再重新评估。",
    )
