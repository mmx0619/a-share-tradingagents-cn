"""A 股风险辩论员。

这个文件对应原版 TradingAgents 的 risk_mgmt 三个角色：

    aggressive_debator.py
    conservative_debator.py
    neutral_debator.py

原版风险流程在 Trader 给出交易提案之后开始：

    Aggressive Analyst
      -> Conservative Analyst
      -> Neutral Analyst
      -> ...
      -> Portfolio Manager

三个角色的分工：

    Aggressive Risk Analyst:
        更愿意接受风险，强调高收益机会。

    Conservative Risk Analyst:
        更重视防守，强调本金安全和波动控制。

    Neutral Risk Analyst:
        在激进和保守之间做平衡。

注意：
    风险辩论员不是重新抓数据。
    它们读取前面已经生成的报告、Research Manager 计划和 Trader 提案，
    然后从不同风险偏好审查交易计划。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


AGGRESSIVE_RISK_PROMPT = """你是 A 股激进风险分析员。

你的角色不是保守避险，而是主动寻找高收益机会。

你需要：

1. 审查 Trader 的交易提案。
2. 强调潜在收益、趋势机会、催化因素和竞争优势。
3. 反驳保守和中性分析员中过度谨慎的部分。
4. 如果 Trader 过于保守，可以说明为什么可以更积极。
5. 如果 Trader 已经很激进，也要指出这种激进是否有数据支持。

你必须基于输入材料，不要编造新行情、新新闻或新财务数据。
输出必须是合法 JSON，不要输出 Markdown，不要加解释文字。
"""


CONSERVATIVE_RISK_PROMPT = """你是 A 股保守风险分析员。

你的首要目标是保护资金、降低波动、避免不必要损失。

你需要：

1. 审查 Trader 的交易提案。
2. 强调潜在亏损、趋势恶化、流动性、基本面缺口、消息面不确定性。
3. 反驳激进和中性分析员中过度乐观的部分。
4. 如果 Trader 给出 Buy，要重点检查入场条件和止损是否充分。
5. 如果 Trader 给出 Sell，也要检查是否存在过度悲观、卖在低点的风险。

你必须基于输入材料，不要编造新行情、新新闻或新财务数据。
输出必须是合法 JSON，不要输出 Markdown，不要加解释文字。
"""


NEUTRAL_RISK_PROMPT = """你是 A 股中性风险分析员。

你的角色是在激进和保守之间寻找平衡。

你需要：

1. 审查 Trader 的交易提案。
2. 同时评估潜在收益和潜在风险。
3. 指出激进观点可能忽视的风险。
4. 指出保守观点可能错过的机会。
5. 给出更稳健、更可执行的风险调整建议。

你必须基于输入材料，不要编造新行情、新新闻或新财务数据。
输出必须是合法 JSON，不要输出 Markdown，不要加解释文字。
"""


class RiskAssessment(BaseModel):
    """风险分析员输出的结构化风险评估。

    role:
        风险分析员角色，必须是 aggressive / conservative / neutral。

    risk_level:
        风险等级，必须是 low / medium / high。

    allow_trade:
        是否允许继续执行 Trader 的交易提案。

    key_risks:
        关键风险列表。

    risk_triggers:
        需要触发风控动作的条件。

    mitigation_plan:
        风险缓释措施。

    position_sizing_advice:
        仓位建议。

    debate_argument:
        该风险角色的核心辩论观点。
    """

    role: Literal["aggressive", "conservative", "neutral"] = Field(description="风险分析员角色。")
    risk_level: Literal["low", "medium", "high"] = Field(description="风险等级。")
    allow_trade: bool = Field(description="是否允许继续执行交易提案。")
    key_risks: list[str] = Field(description="关键风险列表。")
    risk_triggers: list[str] = Field(description="触发风控动作的条件。")
    mitigation_plan: str = Field(description="风险缓释措施。")
    position_sizing_advice: str = Field(description="仓位建议。")
    debate_argument: str = Field(description="该风险角色的核心辩论观点。")


RISK_ASSESSMENT_JSON_REQUIREMENT = """输出要求：

你必须只输出一个合法 JSON 对象，不要输出 Markdown，不要加解释文字。

JSON 格式必须是：

{
  "role": "aggressive | conservative | neutral",
  "risk_level": "low | medium | high",
  "allow_trade": true 或 false,
  "key_risks": ["风险 1", "风险 2"],
  "risk_triggers": ["触发条件 1", "触发条件 2"],
  "mitigation_plan": "风险缓释措施",
  "position_sizing_advice": "仓位建议",
  "debate_argument": "你的核心风险辩论观点"
}

字段要求：

1. role 必须等于当前风险角色。
2. risk_level 只能是 low / medium / high。
3. allow_trade 必须是布尔值。
4. key_risks 和 risk_triggers 必须是字符串数组。
5. 不要编造输入材料里没有的具体价格、新闻、公告或财务数字。
"""


@dataclass
class RiskDebatorContext:
    """风险辩论员上下文。"""

    symbol: str
    trade_date: str
    role: str
    prompt: str


def build_aggressive_risk_context(
    symbol: str,
    trade_date: str,
    market_report: str,
    news_report: str,
    fundamentals_report: str,
    investment_plan: str,
    trader_plan: str,
    risk_history: str = "",
    last_conservative_argument: str = "",
    last_neutral_argument: str = "",
) -> RiskDebatorContext:
    """构造激进风险分析员上下文。"""
    normalized_symbol = normalize_cn_symbol(symbol)
    prompt = build_risk_debator_prompt(
        role_prompt=AGGRESSIVE_RISK_PROMPT,
        symbol=normalized_symbol,
        trade_date=trade_date,
        role="aggressive",
        market_report=market_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        investment_plan=investment_plan,
        trader_plan=trader_plan,
        risk_history=risk_history,
        peer_arguments={
            "上一轮保守风险分析员观点": last_conservative_argument,
            "上一轮中性风险分析员观点": last_neutral_argument,
        },
    )
    return RiskDebatorContext(normalized_symbol, trade_date, "aggressive", prompt)


def build_conservative_risk_context(
    symbol: str,
    trade_date: str,
    market_report: str,
    news_report: str,
    fundamentals_report: str,
    investment_plan: str,
    trader_plan: str,
    risk_history: str = "",
    last_aggressive_argument: str = "",
    last_neutral_argument: str = "",
) -> RiskDebatorContext:
    """构造保守风险分析员上下文。"""
    normalized_symbol = normalize_cn_symbol(symbol)
    prompt = build_risk_debator_prompt(
        role_prompt=CONSERVATIVE_RISK_PROMPT,
        symbol=normalized_symbol,
        trade_date=trade_date,
        role="conservative",
        market_report=market_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        investment_plan=investment_plan,
        trader_plan=trader_plan,
        risk_history=risk_history,
        peer_arguments={
            "上一轮激进风险分析员观点": last_aggressive_argument,
            "上一轮中性风险分析员观点": last_neutral_argument,
        },
    )
    return RiskDebatorContext(normalized_symbol, trade_date, "conservative", prompt)


def build_neutral_risk_context(
    symbol: str,
    trade_date: str,
    market_report: str,
    news_report: str,
    fundamentals_report: str,
    investment_plan: str,
    trader_plan: str,
    risk_history: str = "",
    last_aggressive_argument: str = "",
    last_conservative_argument: str = "",
) -> RiskDebatorContext:
    """构造中性风险分析员上下文。"""
    normalized_symbol = normalize_cn_symbol(symbol)
    prompt = build_risk_debator_prompt(
        role_prompt=NEUTRAL_RISK_PROMPT,
        symbol=normalized_symbol,
        trade_date=trade_date,
        role="neutral",
        market_report=market_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        investment_plan=investment_plan,
        trader_plan=trader_plan,
        risk_history=risk_history,
        peer_arguments={
            "上一轮激进风险分析员观点": last_aggressive_argument,
            "上一轮保守风险分析员观点": last_conservative_argument,
        },
    )
    return RiskDebatorContext(normalized_symbol, trade_date, "neutral", prompt)


def build_risk_debator_prompt(
    role_prompt: str,
    symbol: str,
    trade_date: str,
    role: str,
    market_report: str,
    news_report: str,
    fundamentals_report: str,
    investment_plan: str,
    trader_plan: str,
    risk_history: str,
    peer_arguments: dict[str, str],
) -> str:
    """构造风险辩论员 Prompt。"""
    peer_sections = []
    for title, content in peer_arguments.items():
        peer_sections.append(f"## {title}\n\n{content or '暂无。'}")

    return f"""{role_prompt}

当前分析股票：{symbol}
当前分析日期：{trade_date}
当前风险角色：{role}

## Trader 交易提案

{trader_plan}

## Research Manager 研究计划

{investment_plan}

## Market Agent 技术面报告

{market_report}

## News Agent 新闻面报告

{news_report}

## Fundamentals Agent 基本面报告

{fundamentals_report}

## 当前风险辩论历史

{risk_history or "暂无风险辩论历史。"}

{chr(10).join(peer_sections)}

请基于以上材料，输出你的风险辩论观点。

{RISK_ASSESSMENT_JSON_REQUIREMENT}
"""


def render_risk_debator_context(context: RiskDebatorContext) -> str:
    """渲染风险辩论员上下文。"""
    return context.prompt


def render_risk_assessment(assessment: RiskAssessment) -> str:
    """把结构化风险评估渲染成 Portfolio Manager 可读文本。"""
    allow_text = "允许继续执行交易提案" if assessment.allow_trade else "不建议继续执行交易提案"
    lines = [
        f"**Risk Role**: {assessment.role}",
        f"**Risk Level**: {assessment.risk_level}",
        f"**Allow Trade**: {allow_text}",
        "",
        "**Key Risks**:",
        *[f"- {risk}" for risk in assessment.key_risks],
        "",
        "**Risk Triggers**:",
        *[f"- {trigger}" for trigger in assessment.risk_triggers],
        "",
        f"**Mitigation Plan**: {assessment.mitigation_plan}",
        "",
        f"**Position Sizing Advice**: {assessment.position_sizing_advice}",
        "",
        f"**Debate Argument**: {assessment.debate_argument}",
    ]
    return "\n".join(lines)


def build_fallback_risk_assessment(role: str, error_message: str) -> RiskAssessment:
    """构造风险分析员的保守兜底输出。"""
    normalized_role = role if role in {"aggressive", "conservative", "neutral"} else "neutral"
    return RiskAssessment(
        role=normalized_role,  # type: ignore[arg-type]
        risk_level="high",
        allow_trade=False,
        key_risks=[
            "模型没有返回可校验的风险评估，风险节点进入保守兜底。",
            f"结构化输出错误：{error_message}",
        ],
        risk_triggers=[
            "风险评估输出无法通过程序校验。",
            "关键交易条件无法由结构化字段确认。",
        ],
        mitigation_plan="暂停执行激进交易动作，等待风险节点重新生成合法结构化评估。",
        position_sizing_advice="不新增仓位；如已有仓位，应保持谨慎并等待重新评估。",
        debate_argument="由于风险节点结构化输出失败，本轮风险意见采用保守兜底。",
    )


def risk_assessment_to_json_schema() -> dict:
    """返回 RiskAssessment JSON Schema。"""
    return RiskAssessment.model_json_schema()
