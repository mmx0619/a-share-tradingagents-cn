"""最终交易信号处理层。

原版 TradingAgents 在 Portfolio Manager 输出最终决策后，
还有一层 SignalProcessor：

    final_trade_decision -> 机器可读信号

我们 A 股版也保留这个设计。

为什么需要它？
    报告里的中文结论是给人看的；
    signal 是给程序看的。

例如后续你要做：
    - Web 页面红绿灯；
    - Excel 汇总；
    - 自动记录历史信号；
    - 多只股票对比；

都不应该再去解析长篇报告，而应该直接读取这个稳定字段。

本文件还额外增加一层 A 股版风控护栏：

    Portfolio Manager / Trader / Risk Debate
        -> RiskGuardrailDecision

它不是“市场定理”，也不是保证收益的交易系统。
它是项目里的确定性工程规则：
    当模型给出买入、加仓、观望或卖出时，
    程序再补一层最大仓位、是否允许新增仓位、是否必须等待止损条件的约束。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TradeSignal:
    """机器可读的最终交易信号。

    rating:
        Portfolio Manager 的五档评级：
            Buy / Overweight / Hold / Underweight / Sell

    action:
        程序层面的三档动作：
            BUY / HOLD / SELL

    exposure:
        仓位态度：
            increase / maintain / reduce / exit

    chinese_action:
        给界面或报告显示的中文动作。
    """

    rating: str
    action: str
    exposure: str
    chinese_action: str


@dataclass(frozen=True)
class RiskGuardrailDecision:
    """程序层面的风控护栏决策。

    risk_band:
        风控带：
            normal     常规执行；
            controlled 限制执行；
            defensive  防守执行；
            blocked    禁止新增风险仓位。

    allow_new_position:
        是否允许新开仓。

    allow_add_position:
        是否允许已有仓位继续加仓。

    max_position_pct:
        本次计划允许的目标仓位上限。
        0.05 表示最多 5%。

    max_single_add_pct:
        单次加仓上限。

    required_action:
        程序动作标签，方便以后 Web 页面、Excel 或自动化流程使用。

    chinese_summary:
        给人看的风控摘要。

    reasons:
        为什么这样限制。

    constraints:
        执行前必须满足的约束。
    """

    risk_band: str
    allow_new_position: bool
    allow_add_position: bool
    max_position_pct: float
    max_single_add_pct: float
    required_action: str
    chinese_summary: str
    reasons: list[str]
    constraints: list[str]


def process_portfolio_rating(rating: str) -> TradeSignal:
    """把 Portfolio Manager 五档评级转换成机器可读交易信号。

    映射规则：
        Buy        -> BUY，增加仓位
        Overweight -> BUY，逐步提高仓位
        Hold       -> HOLD，维持/观望
        Underweight-> SELL，降低仓位
        Sell       -> SELL，退出/回避
    """
    normalized = str(rating or "").strip()

    if normalized == "Buy":
        return TradeSignal(
            rating=normalized,
            action="BUY",
            exposure="increase",
            chinese_action="买入或加仓",
        )

    if normalized == "Overweight":
        return TradeSignal(
            rating=normalized,
            action="BUY",
            exposure="increase_gradually",
            chinese_action="逐步提高仓位",
        )

    if normalized == "Hold":
        return TradeSignal(
            rating=normalized,
            action="HOLD",
            exposure="maintain",
            chinese_action="持有或观望",
        )

    if normalized == "Underweight":
        return TradeSignal(
            rating=normalized,
            action="SELL",
            exposure="reduce",
            chinese_action="降低仓位",
        )

    if normalized == "Sell":
        return TradeSignal(
            rating=normalized,
            action="SELL",
            exposure="exit",
            chinese_action="卖出或回避",
        )

    return TradeSignal(
        rating=normalized or "Unknown",
        action="HOLD",
        exposure="unknown",
        chinese_action="无法识别，默认观望",
    )


def build_risk_guardrail_decision(
    trade_signal: TradeSignal,
    trader_action: str,
    trader_position_sizing: str | None = None,
    trader_stop_loss: float | None = None,
    risk_assessments: list[Any] | None = None,
) -> RiskGuardrailDecision:
    """根据最终信号、交易提案和风险辩论生成风控护栏。

    这层规则只做“限制”，不做“增强”：
        如果模型已经说 Sell，这里不会改成 Buy；
        如果风险辩论提示高风险，这里会限制或阻断新增仓位；
        如果 Trader 想 Buy 但没有止损，这里会降低仓位上限。
    """
    assessments = risk_assessments or []
    reasons: list[str] = []
    constraints: list[str] = []

    max_position_pct = base_position_cap_for_rating(trade_signal.rating)
    max_single_add_pct = base_single_add_cap_for_rating(trade_signal.rating)
    allow_new_position = trade_signal.action == "BUY"
    allow_add_position = trade_signal.action == "BUY"
    risk_band = "normal" if trade_signal.action == "BUY" else "blocked"
    required_action = "follow_signal"

    if trade_signal.action == "SELL":
        reasons.append("Portfolio Manager 最终信号是卖出或降低仓位。")
        constraints.append("不允许新增风险仓位；已有仓位应按最终报告考虑降低或退出。")
        return RiskGuardrailDecision(
            risk_band="blocked",
            allow_new_position=False,
            allow_add_position=False,
            max_position_pct=0.0,
            max_single_add_pct=0.0,
            required_action="reduce_or_exit",
            chinese_summary="不允许新增仓位；已有仓位按报告考虑减仓或退出。",
            reasons=reasons,
            constraints=constraints,
        )

    if trade_signal.action == "HOLD":
        reasons.append("Portfolio Manager 最终信号是持有或观望。")
        constraints.append("不主动新开仓；如已有仓位，以观察和复评为主。")
        return RiskGuardrailDecision(
            risk_band="blocked",
            allow_new_position=False,
            allow_add_position=False,
            max_position_pct=0.0,
            max_single_add_pct=0.0,
            required_action="observe_only",
            chinese_summary="观望为主，不新增仓位。",
            reasons=reasons,
            constraints=constraints,
        )

    high_risk_count = count_risk_level(assessments, "high")
    medium_risk_count = count_risk_level(assessments, "medium")
    disallow_count = count_disallow_trade(assessments)
    neutral_blocks = any(
        getattr(assessment, "role", "") == "neutral"
        and getattr(assessment, "allow_trade", True) is False
        for assessment in assessments
    )

    if disallow_count >= 2 or neutral_blocks:
        reasons.append("风险辩论中有多个角色不允许继续交易，或中性风险分析员明确阻断。")
        constraints.append("不允许新增风险仓位，等待重新分析或风险解除。")
        return RiskGuardrailDecision(
            risk_band="blocked",
            allow_new_position=False,
            allow_add_position=False,
            max_position_pct=0.0,
            max_single_add_pct=0.0,
            required_action="block_new_buy",
            chinese_summary="风险辩论阻断，不允许新增仓位。",
            reasons=reasons,
            constraints=constraints,
        )

    if high_risk_count > 0 or disallow_count == 1:
        risk_band = "defensive"
        max_position_pct = min(max_position_pct, 0.05)
        max_single_add_pct = min(max_single_add_pct, 0.02)
        reasons.append("风险辩论中出现 high 风险或单个角色不允许交易。")
        constraints.append("最多小仓位试探，不能一次性重仓。")
    elif medium_risk_count > 0:
        risk_band = "controlled"
        max_position_pct = min(max_position_pct, 0.10)
        max_single_add_pct = min(max_single_add_pct, 0.03)
        reasons.append("风险辩论中出现 medium 风险，需要限制仓位。")
        constraints.append("分批执行，等待行情和风险条件继续确认。")

    model_position_cap = extract_position_pct_cap(trader_position_sizing)
    if model_position_cap is not None:
        max_position_pct = min(max_position_pct, model_position_cap)
        reasons.append(f"Trader 仓位建议中出现 {model_position_cap:.0%} 上限，程序按更保守值执行。")

    if str(trader_action or "").strip() == "Buy" and trader_stop_loss is None:
        risk_band = more_conservative_risk_band(risk_band, "defensive")
        max_position_pct = min(max_position_pct, 0.03)
        max_single_add_pct = min(max_single_add_pct, 0.01)
        reasons.append("Trader 给出 Buy，但没有可校验止损价格。")
        constraints.append("补充止损或失效条件前，只允许观察或极小仓位。")

    if not reasons:
        reasons.append("最终信号允许买入，且风险辩论没有触发额外限制。")

    if not constraints:
        constraints.append("按报告分批执行，不追高，不超过风控仓位上限。")

    required_action = required_action_for_band(risk_band)
    chinese_summary = build_guardrail_summary(
        risk_band=risk_band,
        max_position_pct=max_position_pct,
        max_single_add_pct=max_single_add_pct,
    )
    return RiskGuardrailDecision(
        risk_band=risk_band,
        allow_new_position=max_position_pct > 0,
        allow_add_position=max_single_add_pct > 0,
        max_position_pct=round(max_position_pct, 4),
        max_single_add_pct=round(max_single_add_pct, 4),
        required_action=required_action,
        chinese_summary=chinese_summary,
        reasons=reasons,
        constraints=constraints,
    )


def base_position_cap_for_rating(rating: str) -> float:
    """根据 Portfolio Manager 评级给出初始仓位上限。"""
    caps = {
        "Buy": 0.20,
        "Overweight": 0.15,
        "Hold": 0.0,
        "Underweight": 0.0,
        "Sell": 0.0,
    }
    return caps.get(str(rating or "").strip(), 0.0)


def base_single_add_cap_for_rating(rating: str) -> float:
    """根据 Portfolio Manager 评级给出单次加仓上限。"""
    caps = {
        "Buy": 0.05,
        "Overweight": 0.03,
        "Hold": 0.0,
        "Underweight": 0.0,
        "Sell": 0.0,
    }
    return caps.get(str(rating or "").strip(), 0.0)


def count_risk_level(assessments: list[Any], risk_level: str) -> int:
    """统计某个风险等级出现次数。"""
    return sum(
        1
        for assessment in assessments
        if str(getattr(assessment, "risk_level", "")).strip() == risk_level
    )


def count_disallow_trade(assessments: list[Any]) -> int:
    """统计不允许交易的风险分析员数量。"""
    return sum(
        1
        for assessment in assessments
        if getattr(assessment, "allow_trade", True) is False
    )


def extract_position_pct_cap(text: str | None) -> float | None:
    """从仓位建议文本中提取百分比或几成仓位。

    示例：
        "5% 以内" -> 0.05
        "不超过两成" -> 0.20
    """
    if not text:
        return None

    source = str(text)
    percent_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", source)
    percent_values = [float(value) / 100 for value in percent_matches if float(value) >= 0]
    chinese_cap = extract_chinese_cheng_cap(source)

    values = percent_values
    if chinese_cap is not None:
        values.append(chinese_cap)

    if not values:
        return None
    return min(values)


def extract_chinese_cheng_cap(text: str) -> float | None:
    """提取“一成、两成、三成”这类 A 股常见仓位表述。"""
    mapping = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    match = re.search(r"([一二两三四五六七八九十]|\d+(?:\.\d+)?)\s*成", text)
    if not match:
        return None

    raw_value = match.group(1)
    if raw_value in mapping:
        value = mapping[raw_value]
    else:
        value = float(raw_value)
    return value / 10


def more_conservative_risk_band(current: str, candidate: str) -> str:
    """返回更保守的风控带。"""
    order = {
        "normal": 0,
        "controlled": 1,
        "defensive": 2,
        "blocked": 3,
    }
    return current if order.get(current, 0) >= order.get(candidate, 0) else candidate


def required_action_for_band(risk_band: str) -> str:
    """根据风控带返回程序动作标签。"""
    mapping = {
        "normal": "allow_planned_buy",
        "controlled": "limit_position",
        "defensive": "small_probe_only",
        "blocked": "block_new_buy",
    }
    return mapping.get(risk_band, "observe_only")


def build_guardrail_summary(
    risk_band: str,
    max_position_pct: float,
    max_single_add_pct: float,
) -> str:
    """生成给人看的风控摘要。"""
    if risk_band == "normal":
        return (
            f"允许按计划分批执行，目标仓位不超过 {max_position_pct:.0%}，"
            f"单次加仓不超过 {max_single_add_pct:.0%}。"
        )
    if risk_band == "controlled":
        return (
            f"允许受控小仓位参与，目标仓位不超过 {max_position_pct:.0%}，"
            f"单次加仓不超过 {max_single_add_pct:.0%}。"
        )
    if risk_band == "defensive":
        return (
            f"只允许防守性试探，目标仓位不超过 {max_position_pct:.0%}，"
            f"单次加仓不超过 {max_single_add_pct:.0%}。"
        )
    return "不允许新增仓位。"


def render_risk_guardrail_decision(decision: RiskGuardrailDecision) -> str:
    """把风控护栏渲染成 Markdown。"""
    allow_new = "是" if decision.allow_new_position else "否"
    allow_add = "是" if decision.allow_add_position else "否"
    lines = [
        f"**Risk Band**: {decision.risk_band}",
        f"**Required Action**: {decision.required_action}",
        f"**Allow New Position**: {allow_new}",
        f"**Allow Add Position**: {allow_add}",
        f"**Max Position Pct**: {decision.max_position_pct:.0%}",
        f"**Max Single Add Pct**: {decision.max_single_add_pct:.0%}",
        "",
        f"**Summary**: {decision.chinese_summary}",
        "",
        "**Reasons**:",
        *[f"- {reason}" for reason in decision.reasons],
        "",
        "**Constraints**:",
        *[f"- {constraint}" for constraint in decision.constraints],
    ]
    return "\n".join(lines)
